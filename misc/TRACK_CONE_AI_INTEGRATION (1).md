# Track / Cone / AI-Driving Integration — Implementation Instructions

This document specifies how to add a new feature to the suspension-tuning app
(`MAIN.py`): load a track definition from a CSV or JSON file, spawn cones
along it on a flat BeamNG map, drive the centerline with BeamNG's built-in AI
(no `BeamNG.tech` license), and poll/log/present every channel of telemetry
that a plain **BeamNG.drive** license actually exposes.

Read `HANDOFF.md` first — this feature reuses the existing BeamNG bridge
(`load_beamng_scenario`, `run_beamng_test`, the version-compat shim, the
"don't trust prefab-embedded spawns, spawn live at runtime instead" pattern)
rather than replacing it. Everything below is written to slot into that
existing bridge, not to build a parallel one.

---

## 0. License reality check (do this before writing code)

`beamngpy`'s own docs are written against `BeamNG.tech`, and quietly assume
features that don't exist on a plain Steam/GOG `BeamNG.drive` license. The
existing code already hit this once (`AdvancedIMU` → server-side rejection).
Confirm the following **in your own installed game build** before relying on
any of it, because BeamNG changes sensor/feature licensing between versions:

| Capability | Status on plain BeamNG.drive | Source |
|---|---|---|
| `State` sensor (pos, dir, up, vel, front, quat) | Free | already used in `run_beamng_test` |
| `Electrics` sensor (wheel speed, RPM, throttle/brake/steering, lights, etc.) | Free | already used in `load_beamng_scenario` |
| `Timer` sensor | Free | documented as a plain Python-API sensor, no tech gate |
| `Damage` sensor (part damage / deformation) | Free | documented as a plain Python-API sensor, no tech gate |
| `GForces` sensor (vehicle-frame linear acceleration, raw + smoothed) | Free | documented as a plain Python-API sensor, no tech gate |
| `AdvancedIMU` | **Tech-only** | already confirmed by this project — rejected server-side |
| `Camera` / `Lidar` / `Ultrasonic` / `IdealRadar` / `Mesh` / `GPS` / `RoadsSensor` | Unconfirmed — treat as tech-only until verified | BeamNGpy's docs describe them in the BeamNG.tech context; some may be gated, some may just be heavy |
| `vehicle.ai.execute_script()` (time-stamped waypoint list → AI drives it) | Free — this is a base-game "Script AI" feature, present since ~v0.20, used for scripted traffic/crash scenarios in vanilla BeamNG.drive | in-game AI Script Manager predates BeamNG.tech |
| `vehicle.ai.drive_using_waypoints()` / `ai.set_mode('span'/'traffic'/'flee'/'chase')` (navgraph-based) | Free in principle, but **useless here** — `MAIN.py` already found `ai.set_mode('speed')` silently does nothing off a navgraph road, and a flat/procedural map has no navgraph | existing code comment in `run_beamng_test` |
| `ScenarioObject` / `StaticObject` (place non-vehicle meshes/props) | Free — core scenario authoring feature | BeamNGpy `scenario` module |
| Live runtime prop spawn (same technique `load_beamng_scenario` uses for the vehicle, i.e. bypass the prefab) | Free, same mechanism as the existing vehicle-spawn workaround | see §3 |

**Action item before coding:** open the in-game vehicle/prop selector (close
the World Editor if open, then use the vehicle selector — props are listed
after vehicles) and find the exact internal name of a cone-shaped prop
(commonly a small traffic-cone or pylon mesh shipped with the base game).
Do not guess the asset ID in code — different BeamNG versions have shipped
different prop sets, and spawning a wrong/missing ID fails silently or
throws late in `scenario.make()`. Put the confirmed name in a new
`BEAMNG_CONE_PROP_MODEL` constant (see §3) once you have it.

---

## 1. Track file format

Support both CSV and JSON with the same logical schema so the parser can be
one function with two loaders feeding a shared validator.

### Required fields (per track point, in order along the track)

- `x`, `y` — world-space coordinates, meters, same coordinate frame as
  `BEAMNG_SPAWN_POS` (i.e. whatever frame the target level uses).
- `z` — optional. If omitted, resolve height by raycast/terrain query at
  spawn time (see §2); if the target map is a genuinely flat plane (e.g.
  `gridmap_v2`), a constant fallback `z` is fine.

### Optional fields

- `width_m` — track width at this point (cones placed `width_m / 2` either
  side of the centerline). If omitted, fall back to a global
  `DEFAULT_TRACK_WIDTH_M` constant.
- `left_x`, `left_y`, `right_x`, `right_y` — explicit boundary points, if the
  source data already has them (e.g. exported from a track-mapping tool).
  When present, these override `width_m`-based offset for that point.
- `speed_mps` — target AI speed at this point. If omitted, derive from
  local curvature (see §4) or fall back to a flat `DEFAULT_TRACK_SPEED_MPS`.
- `cone_spacing_m` — override the default cone spacing for this segment.

### CSV example

```csv
x,y,z,width_m,speed_mps
-337.6,-491.3,100.15,8.0,12
-320.1,-505.4,100.10,8.0,14
-298.7,-522.9,100.05,7.5,16
...
```

### JSON example

```json
{
  "name": "gridmap_autocross_1",
  "map": "gridmap_v2",
  "closed_loop": true,
  "default_width_m": 8.0,
  "default_speed_mps": 14.0,
  "points": [
    {"x": -337.6, "y": -491.3, "z": 100.15, "width_m": 8.0, "speed_mps": 12},
    {"x": -320.1, "y": -505.4, "z": 100.10, "width_m": 8.0, "speed_mps": 14}
  ]
}
```

### Parsing/validation rules

- Minimum 3 points (this is also `execute_script`'s hard minimum for a
  valid AI script — see §4 — so enforce it at parse time with a clear error
  rather than letting it fail deep inside the BeamNG bridge).
- Reject/flag consecutive points closer than some minimum spacing (e.g.
  0.5 m) — near-duplicate points produce a near-zero-duration script
  segment, which `execute_script` explicitly cannot satisfy physically
  (see the "AI cannot drive along scripts that are physically impossible"
  note in §4) and it will start skipping points.
- If `closed_loop` is true, either require `points[0] == points[-1]` or
  synthesize the closing point yourself (documented in the loader either
  way — don't silently do one or the other).
- Store the parsed result as a single internal structure, e.g. a list of
  `TrackPoint` dataclasses (`x, y, z, width_m, speed_mps`), so cone
  placement (§2), AI script generation (§4), and the cross-track-error
  calculation (§5) all consume the same normalized representation instead
  of re-parsing the file three different ways.

---

## 2. Deriving cone (boundary) positions from the centerline

For each `TrackPoint` that doesn't already have explicit `left_*`/`right_*`
coordinates:

1. Compute the local tangent direction using neighboring points (central
   difference for interior points, forward/backward difference at the
   open ends unless `closed_loop`, in which case wrap around).
2. Rotate the tangent 90° in the XY plane to get the left/right normal.
3. Offset by `width_m / 2` along that normal to get left- and
   right-boundary points.
4. Walk along the boundary (not the raw point list) and drop a cone every
   `cone_spacing_m` (global default, overridable per-point) using arc-length
   resampling — not "one cone per input point," since input point density
   has nothing to do with desired cone density and the two should be
   decoupled.
5. Resolve each cone's `z`: if the source point didn't supply one, either
   reuse the centerline point's interpolated `z`, or (more robust, works on
   non-flat maps too) do it live in-game — spawn slightly above the
   expected height and let BeamNG's physics settle the prop onto the
   terrain, or use whatever terrain-height query `beamngpy`'s scenario/env
   API exposes in your installed version (check `beamng_client.env` /
   `beamng_client.scenario` for a height query method — this varies by
   `beamngpy` version, confirm against your installed one rather than the
   latest docs).

Output of this stage: a flat list of `(x, y, z, side)` cone spawn poses
(`side` = `"left"`/`"right"`, useful later for scoring "which boundary did
the car cross").

---

## 3. Spawning the cones in BeamNG

`HANDOFF.md` already documents why this project does **not** rely on
prefab-embedded objects: on the installed build, prefab-embedded vehicles
failed to spawn and the fix was to load an empty scenario, then spawn the
vehicle live via `beamng_client.vehicles.spawn(...)`. Apply the same
distrust to cone props — prefer a live/runtime spawn path over baking cones
into the prefab, and treat "cones show up in the prefab file but not in the
running game" as the default failure mode to watch for, exactly like the
vehicle case.

Two viable approaches, in order of preference:

### 3a. `ScenarioObject`/`StaticObject`, added before `scenario.make()`

```python
from beamngpy.scenario import StaticObject  # class name/import path may
                                              # differ by beamngpy version —
                                              # confirm against your installed
                                              # version's `scenario` module

for i, (cx, cy, cz, side) in enumerate(cone_positions):
    cone = StaticObject(
        name=f"track_cone_{i}",
        pos=(cx, cy, cz),
        rot_quat=(0, 0, 0, 1),
        scale=(1, 1, 1),
        shape=BEAMNG_CONE_PROP_MODEL,  # confirmed asset name from §0
    )
    scenario.add_object(cone)
```

This is the "official" path and is simplest, but given this project's
existing experience with prefab-embedded objects silently failing to
appear on this specific BeamNG build, **verify it actually shows up
in-game before building anything else on top of it.** If it exhibits the
same failure as the prefab vehicle did, fall back to 3b.

### 3b. Live runtime spawn (same pattern as the existing vehicle workaround)

If 3a's prefab-baked cones don't reliably appear (mirroring the vehicle
issue), spawn them live after `scenario.start()`, the same way
`load_beamng_scenario` spawns `test_vehicle` live via
`beamng_client.vehicles.spawn(...)` instead of `scenario.add_vehicle(...)`.
Props/statics generally don't have a dedicated `.spawn()` convenience like
vehicles do — the practical route is `beamng_client.control.queue_lua_command()`
(or your installed `beamngpy` version's equivalent raw-Lua-call API) to call
the in-game console-level spawn function directly, e.g. the same
`core_vehicles.spawnNewVehicle("<prop_id>", ...)` mechanism the game's own
console/vehicle-selector uses for props (props are spawned through the
vehicle-spawn system in BeamNG, not a separate "prop system" — this is why
the in-game vehicle selector lists props after vehicles, per §0). Confirm
the exact Lua call signature against your installed game version in the
in-game Lua console before wiring it into Python — this is exactly the kind
of thing that already drifted once in this project (`GetCurrentVehicles`
shape change) and could again.

Whichever path you use, log an explicit count check after spawning
(`beamng_client.vehicles.get_current()` won't include static props, so use
whatever "objects in world" query is present for your `beamngpy` version —
`scenario.find_procedural_meshes()`/`find_static` per the API docs, or
similar) and surface "requested N cones, found M in-world" in the app, the
same way `load_beamng_scenario` already checks for the `.prefab`/`.json`
files actually landing on disk before proceeding.

---

## 4. Turning the centerline into an AI-drivable path

Use `vehicle.ai.execute_script()` — this is the correct tool for "arbitrary
polyline on a map with no navgraph," which is exactly this project's
existing situation (`run_beamng_test` already notes `ai.set_mode('speed')`
requires navgraph-recognized roads and falls back to raw throttle/steering
for that reason). `execute_script` does not need a navgraph; it drives to
timestamped positions directly.

### Building the script

`execute_script` takes a list of dicts, each with a `pos` (`x, y, z`) and a
timestamp; speed is implicit in the spacing between timestamps, not passed
per-point (unlike `drive_using_waypoints`'s `speed` field on the polyline
method — don't mix the two APIs' point formats up).

```python
def build_ai_script(track_points, closed_loop):
    """track_points: parsed TrackPoints in centerline order.
    Returns a list of {'x', 'y', 'z', 't'} dicts for vehicle.ai.execute_script().
    """
    script = []
    t = 0.0
    pts = track_points + [track_points[0]] if closed_loop else track_points
    for prev, cur in zip(pts, pts[1:]):
        dist = ((cur.x - prev.x) ** 2 + (cur.y - prev.y) ** 2) ** 0.5
        speed = cur.speed_mps or DEFAULT_TRACK_SPEED_MPS
        dt = max(dist / max(speed, 0.1), MIN_SCRIPT_DT)  # avoid a
        # near-zero dt between close points, which is the same physically-
        # impossible-segment problem flagged in the parser (§1) — clamp
        # here too as a second line of defense
        t += dt
        script.append({"x": cur.x, "y": cur.y, "z": cur.z, "t": t})
    return script
```

`execute_script` requires **at least 3 nodes** (matches the §1 parser
minimum) and its own docs are explicit that it does its best, not a
guarantee: "cannot drive along scripts that would be physically
impossible... if the AI falls behind schedule, it will start skipping
points." Practically:

- Don't feed it a raw one-cone-spacing-resolution polyline (too many nodes,
  too little curvature per node) — resample the *centerline* (not the cone
  list) at a coarser, curvature-aware interval for the script itself. A
  reasonable starting point: fixed arc-length resampling (e.g. every 3–5 m)
  plus extra nodes inserted at high-curvature points so corners aren't
  cut. Keep it separate from cone spacing (§2), which is a visual/track
  concern, not an AI-following concern.
- If a `speed_mps` wasn't supplied per-point, derive one from local
  curvature (tighter turn → lower target speed) rather than a flat
  constant, or the AI will be asked to hold full speed through corners
  and will fall behind schedule and start skipping nodes exactly as the
  docs warn.

### Running it

```python
test_vehicle.ai.execute_script(script, cling=True, start_delay=0.0)
```

`cling=True` snaps the initial vehicle placement onto the script's first
node/orientation — keep this on unless you have a reason to preserve the
exact spawn pose from `BEAMNG_SPAWN_POS`.

---

## 5. Polling loop and telemetry

Extend `run_beamng_test`'s existing poll loop rather than writing a second
one — same `test_vehicle.sensors.poll()` → read `state`/`electrics` pattern,
same stale-connection retry counter, same telemetry-dict-of-lists
accumulation style. Add:

- **`Timer` sensor** — attach alongside `Electrics`; poll for lap/segment
  timing instead of hand-rolling elapsed-time math from wall-clock
  `time.time()` calls (`run_beamng_test` currently starts its own timer at
  the top of the function — the `Timer` sensor is the in-sim equivalent
  and is pause-aware, per its docs, which wall-clock timing is not).
- **`GForces` sensor** — attach it. This directly replaces the existing
  finite-differenced vertical-velocity workaround (`state['vel'][2]`
  between polls) that `run_beamng_test` uses specifically *because*
  `AdvancedIMU` is tech-gated. `GForces` reports linear acceleration in the
  vehicle frame natively and is free — it's a strictly better free
  substitute for what the existing bump-test code is approximating by
  hand, and should be considered for backporting into the existing
  suspension-test path too, not just the new track feature.
- **`Damage` sensor** — attach it. This is the only free way to detect
  "the car hit something" (a cone, a wall) — there is no dedicated free
  "collision/contact" event sensor, so treat any nonzero delta in the
  `Damage` sensor's part-damage readings during a run as a proxy for
  "contact occurred," and cross-reference the vehicle's position at that
  timestamp against the cone list (§2) to guess which cone was likely hit
  (nearest cone within some threshold at that timestamp). This is an
  approximation, not a ground-truth "hit cone #7" event — document it as
  such in the UI rather than presenting it as exact.
- **Cross-track error** — not a sensor at all, compute it yourself each
  poll: project the polled `state['pos']` onto the nearest segment of the
  centerline (simple closest-point-on-polyline search over the
  `TrackPoint` list from §1) and record signed lateral distance from
  centerline plus distance-traveled-along-track. This is what actually
  tells you "how well did the AI follow the line," which is the real
  point of building the track feature at all — raw x/y/z alone doesn't
  answer that.
- **Existing channels** — keep everything `run_beamng_test` already
  collects (`pos_x/y/z`, wheel angular velocity from `Electrics` where
  available, speed error against target).

Append one row per poll to a telemetry structure keyed the same way the
existing one is (dict of lists, one list per channel) so it can reuse
whatever plotting/export code already exists for the suspension-bump test,
rather than inventing a second telemetry shape.

---

## 6. Presenting the data (Streamlit)

Match the app's existing look/feel — this is a mode alongside the
suspension-bump validation, not a separate app.

- **Track map** — 2D matplotlib plot: centerline (line), left/right cones
  (scatter, colored by side), actual driven path (line, from polled
  `pos_x/pos_y`), color-coded by speed or by cross-track error magnitude
  so "where did it struggle" is visible at a glance. Mark any inferred
  cone-contact points (§5) distinctly.
- **Time-series panel** (reuse the existing bump-test chart style):
  speed vs. target speed, cross-track error, `GForces` lateral/longitudinal
  components, `Damage` delta.
- **Lap summary** — total time (`Timer` sensor), average/peak cross-track
  error, average/peak lateral G, inferred cone-contact count, whether the
  script completed all nodes vs. got cut short (compare final polled
  distance-along-track to the script's total length — a shortfall signals
  the AI fell behind schedule and skipped nodes, per §4's documented
  behavior, which is itself worth surfacing to the user as a warning
  rather than silently truncating the plot).
- **CSV export** of the full per-poll telemetry table — the app already
  has no persistent run history (per `HANDOFF.md`'s open items), so don't
  invent a database; a "download telemetry as CSV" button is enough and
  matches the project's current scope.

---

## 7. Suggested new constants (put near the existing `BEAMNG_*` block)

```python
BEAMNG_CONE_PROP_MODEL = "<confirmed asset name from §0 — do not guess>"
DEFAULT_TRACK_WIDTH_M = 8.0
DEFAULT_TRACK_SPEED_MPS = 14.0
DEFAULT_CONE_SPACING_M = 5.0
MIN_TRACK_POINT_SPACING_M = 0.5
MIN_SCRIPT_DT = 0.05  # seconds; floor for execute_script's per-node timing
```

---

## 8. Verification plan (match this project's existing ad hoc style)

There's still no `pytest` suite (see `HANDOFF.md`'s open items) — until one
exists, verify this feature the same way the suspension cost-function
changes were verified: extract the pure-logic pieces (track parsing, cone
offset/resampling math, `execute_script` timestamp generation,
closest-point-on-polyline cross-track math) into a standalone script with
no Streamlit/BeamNG dependency, and check them against hand-computed cases
before ever touching the live game:

- A straight track segment with known width → hand-verify cone left/right
  offsets land where expected.
- A closed-loop square/rectangle track → verify cone spacing wraps cleanly
  across the closing segment with no double cone / gap at the seam.
- A synthetic constant-speed straight line → verify `build_ai_script`'s
  timestamps match `distance / speed` by hand.
- A synthetic point on/near a known polyline → verify the cross-track-error
  function returns ~0 for on-line points and the correct signed distance
  for known off-line offsets.

Only after those pass standalone, run the in-game smoke test: small
(3–5 point) test track on the target flat map, confirm cones appear where
expected (screenshot/F11 free-cam check), confirm `execute_script` completes
without falling obviously behind schedule on a short/slow track, confirm all
new sensors (`Timer`, `GForces`, `Damage`) poll without the "This feature
requires a BeamNG.tech license" error before building the full pipeline on
top of them.

---

## 9. Open items to flag for whoever implements this

- §0 and §3's asset-name and Lua-call details are placeholders by design —
  they depend on the exact installed BeamNG/`beamngpy` versions and must be
  confirmed in-game, the same way the existing `_patched_get_current_info`
  shim exists because of a version-specific API mismatch. Don't skip that
  verification step to save time; this project has already been bitten
  once by assuming `beamngpy`'s docs (written for `BeamNG.tech`) apply
  as-is to a plain `BeamNG.drive` install.
- Cone-hit detection (§5) is inferred from `Damage` deltas + nearest-cone
  distance, not a true collision event — a free-license limitation, not an
  oversight. If precise per-cone hit detection becomes a hard requirement,
  that likely needs `BeamNG.tech` (dedicated contact/collision sensors) and
  is out of scope here.
- Terrain-height resolution for cones/track points without an explicit `z`
  (§2 step 5) is version-dependent and left unresolved above — pin down
  whichever height-query mechanism your installed `beamngpy` actually
  exposes before relying on it, or require `z` in the input file and skip
  the problem entirely for the first version of this feature.
- `execute_script`'s "falls behind schedule → skips points" behavior means
  a badly-tuned speed profile can silently produce a shortcut lap rather
  than an error. §6's "script completion" check is a mitigation, not a
  guarantee — treat any run with a shortfall as suspect rather than trusting
  its telemetry at face value.
