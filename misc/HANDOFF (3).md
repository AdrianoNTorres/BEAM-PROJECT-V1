# Project Handoff — Suspension Tuning Advisor

This file is for whoever (human or agent) picks this project up next. It
covers what the app does, what was just changed, why, and what's still
open. Read this before touching `MAIN.py`.

## What this project is

A Streamlit app (`MAIN.py`, single file) that:

1. Simulates a half-car (pitch-plane) model driving over a speed bump —
   `HalfCarPitchSimulator`, an 8-state ODE integrated with `solve_ivp`.
2. Runs Bayesian optimization (Gaussian process + LCB acquisition) over
   front/rear spring rate and damping to minimize a cost function that
   represents ride harshness.
3. Optionally validates the winning setup in BeamNG.drive via `beamngpy`,
   comparing simulated vs. real telemetry.

Full project description and usage: see `README.md` — it's accurate as of
this handoff and grounded in the actual code (not aspirational).

## Most recent work: Track/cone AI-driving mode (new feature, implemented but not yet run in-game)

Added a second, separate BeamNG mode alongside the existing suspension-bump
validation: load a track from a CSV/JSON centerline file, spawn cones along
its boundary, drive it with BeamNG's own AI (`vehicle.ai.execute_script()`),
and poll/plot/export every free-license telemetry channel available while
doing so. Full design rationale is in the new `TRACK_CONE_AI_INTEGRATION.md`
— read that alongside this section, it's not duplicated here.

**Everything is additive.** No existing suspension-bump code path
(`HalfCarPitchSimulator`, the BO loop, `load_beamng_scenario`/
`run_beamng_test`) was touched. The new feature lives in its own sections of
`MAIN.py` (`1.5 TRACK / CONE PATH GEOMETRY`, `2.5 TRACK / CONE AI-DRIVING
BEAMNG BRIDGE`, and a new `🏁 Track / Cone AI Driving (Beta)` expander in the
UI) and its own `st.session_state` keys (`track_bng_client`,
`track_bng_vehicle`, `parsed_track`, `track_scenario_result`,
`last_track_run_result`) so it can't collide with the suspension test's
`active_bng_client`/`bng_scenario_loaded` state. The two BeamNG sessions are
independent but not meant to run concurrently — the UI doesn't currently stop
you from opening both, and doing so will fight over `BEAMNG_PORT`/the single
game instance. Worth adding an explicit guard for that if this gets used
regularly.

**What was implemented, by section:**

- **`1.5` — pure track/cone/path geometry, no Streamlit or beamngpy
  dependency.** `TrackPoint`/`TrackDefinition` dataclasses; `load_track_file`
  (+ `parse_track_csv`/`parse_track_json`) with validation (≥3 points after
  near-duplicate removal, matching `execute_script`'s own hard minimum);
  `compute_boundary_offsets` (tangent → normal → width/2 offset per point);
  `generate_cone_positions` (arc-length resample each boundary independently
  of input point density); `build_ai_script` (coarser arc-length resample of
  the *centerline*, converted to `execute_script`'s timestamped-waypoint
  format, with a `MIN_SCRIPT_DT` floor so near-duplicate-speed segments can't
  produce a near-zero `dt`); `closest_point_on_polyline` (cross-track error —
  signed lateral offset + distance-along-track via nearest-segment
  projection).
- **`2.5` — BeamNG bridge.** `load_beamng_track_scenario` mirrors
  `load_beamng_scenario`'s "empty scenario + live vehicle spawn" structure
  exactly, and additionally tries to bake cones in as `StaticObject`s before
  `scenario.make()`. `run_beamng_track_test` calls
  `test_vehicle.ai.execute_script()` on the built script and polls `state`,
  `electrics`, `timer`, `gforces`, and `damage` each tick, computing
  cross-track error live and inferring possible cone contacts from `damage`
  deltas + nearest-cone distance (explicitly a heuristic, not a real
  collision event — see `TRACK_CONE_AI_INTEGRATION.md` §5).
- **UI** — file uploader, track/cone preview plot (uses the same geometry
  functions the run itself will use, so the preview can't drift from what
  actually gets driven), load/run/terminate buttons, driven-path-vs-track
  plot colored by cross-track error, a 3-panel time-series (speed,
  cross-track error, lateral-G + damage), CSV telemetry export, and an
  inferred-cone-hit table.

### What's unverified — read this before running it

This was built and verified as far as it *can* be verified without a running
BeamNG instance or network access in the environment it was written in (no
`beamngpy`, no `streamlit`, both installs blocked — see "How this was
verified" below for exactly what that means in practice). Specifically still
open:

- **`BEAMNG_CONE_PROP_MODEL` is `None`.** This is deliberate, not an
  oversight — see `TRACK_CONE_AI_INTEGRATION.md` §0. Cone spawning will fail
  fast with a clear Streamlit warning/error until someone opens the in-game
  vehicle/prop selector, finds the real cone/pylon prop name for the
  installed BeamNG version, and sets this constant. Track parsing/preview
  and (cone-less) AI driving still work without it.
- **Cone placement path (`StaticObject` + `scenario.add_object`, added
  before `scenario.make()`) is untested against this build.** `HANDOFF.md`'s
  own prior entries already document this exact BeamNG build failing to
  spawn a *prefab-embedded vehicle* for what looks like a prefab-schema
  mismatch — cones going in via the same prefab-bake mechanism could hit the
  same wall. `load_beamng_track_scenario` treats a cone-placement exception
  as a non-fatal warning (`cone_warning` in its result) rather than aborting
  the whole scenario, and separately tries to report a confirmed in-world
  cone count after load (via `find_static_objects`/`find_objects_class`,
  whichever the installed `beamngpy` exposes) — but if cones silently don't
  appear the same way the vehicle didn't, the fallback described in
  `TRACK_CONE_AI_INTEGRATION.md` §3b (live runtime spawn via a raw Lua call,
  the same category of workaround as the vehicle's `vehicles.spawn()` fix)
  still needs to be written. Start there if `cones_confirmed` comes back 0
  or `"unknown"` with cones visibly missing in-game.
- **`Timer`/`GForces`/`Damage` sensor field names are best-effort.**
  `run_beamng_track_test` reads `gforces_data.get('gx', gforces_data.get('gx2', 0.0))`-style
  fallbacks because the exact dict keys these sensors return aren't
  confirmed against the installed `beamngpy` version (only `Electrics` and
  `state` have been exercised live in this project so far, per the existing
  bump-test code). If a poll comes back with all-zero G-forces or an
  always-zero damage total, check the actual dict shape via a quick
  `print(test_vehicle.gforces)` /`print(test_vehicle.damage)` in-game before
  assuming the run itself is broken.
- **`GForces` as a `vertical_g`/IMU replacement was flagged but not
  backported.** `TRACK_CONE_AI_INTEGRATION.md` §5 notes the existing
  suspension bump-test's finite-differenced `vertical_g` (worked around
  `AdvancedIMU` being tech-gated) could likely be replaced by the free
  `GForces` sensor directly. Not done here to keep this change additive —
  worth a follow-up once `GForces`'s real field names are confirmed above,
  since it'd simplify `run_beamng_test` too.
- **AppTest / live Streamlit run wasn't possible in this environment** —
  `pip install streamlit` failed (no network egress in the sandbox this was
  written in). Syntax (`py_compile`) and `ast.parse` both pass, and the full
  new UI block was read back carefully against the existing `elif
  scenario_ready` / button-state patterns in the suspension-test UI, but it
  has not actually been rendered. Run `streamlit run MAIN.py` once before
  trusting the new expander's control flow (e.g. the Load→Run→Terminate
  button visibility logic) end-to-end.

### How this was verified

No `pytest` suite exists yet (same "Open items" note as every prior entry
here). Given no `beamngpy`/`streamlit` available in this environment, the
verification that *was* possible mirrors this project's standard
standalone-extraction technique, scoped to the parts that don't need either
dependency:

- Extracted every function in the new `1.5` section into a standalone
  script (no Streamlit/beamngpy imports) and ran it against the same style
  of hand-computed cases `TRACK_CONE_AI_INTEGRATION.md` §8 lays out:
  - A straight 8 m-wide track → `compute_boundary_offsets` placed left/right
    points at exactly `y = ±4` for a straight line along `y=0`, as expected.
  - A closed-loop 20×20 square track, 6 m wide, 5 m cone spacing → cone
    count came back consistent with (perimeter / spacing) × 2 sides with no
    exception or obviously duplicated/missing seam.
  - A synthetic 3-point straight line at a constant 10 m/s → `build_ai_script`
    produced exact `t = distance / speed` timestamps (`t=5.0` at 50 m,
    `t=20.0` at 200 m).
  - `closest_point_on_polyline` returned `lateral ≈ 0` for an on-line point
    and the exact known offset (`3.0`) for a point 3 m to the left of a
    straight track, with the correct sign convention (matches
    `compute_boundary_offsets`'s left-hand-normal convention).
  - CSV and JSON parsing round-tripped correctly through `load_track_file`,
    including `closed_loop`/`default_width_m`/`default_speed_mps` from a
    JSON file's top-level fields.
  - Confirmed `load_track_file` raises a clear `ValueError` (not a crash)
    both for a file with fewer than 3 points and for a file that drops below
    3 points *after* near-duplicate-point removal.
- `python3 -m py_compile MAIN.py` and `ast.parse` both pass on the full
  file.
- Did **not** verify: anything requiring `beamngpy` or a live BeamNG
  instance (scenario loading, cone spawning, `execute_script` behavior,
  actual sensor field shapes), or anything requiring `streamlit` (the new UI
  block's actual rendering/control flow). See "What's unverified" above —
  that section is the real next-steps list for this entry, more so than
  "Open items" below.

## Earlier work: Custom profile gets adjustable search bounds + per-term toggles

Two additions to the "Custom" objective in the sidebar, on top of the
existing per-term weight number inputs:

- **Adjustable spring/damping search range.** New "Suspension search range"
  number inputs (Front/Rear × Spring/Damping × Min/Max, 8 inputs total)
  inside the Custom expander override `FRONT_SPRING_MIN/MAX`, etc. for that
  run - both the manual sliders below and the Bayesian Optimization loop's
  sampling bounds and `L-BFGS-B` bounds. Guardrails:
  - Absolute limits (`CUSTOM_SPRING_RANGE_ABSOLUTE_LIMITS`,
    `CUSTOM_DAMPING_RANGE_ABSOLUTE_LIMITS`, near the other top-of-file
    constants) cap how far these can be pushed - in particular, spring rate
    can't reach 0, which would divide by zero in
    `compute_damping_ratios()`.
  - If a user sets min >= max on any axis, `_validated_range()` (defined
    inline in the Custom block) shows a warning and falls back to that
    axis's global default range rather than letting an inverted/empty range
    reach the sliders or the optimizer.
  - Any already-set slider value is clamped into the new range right before
    the sliders are drawn (`st.session_state.fs_rate = min(max(...))`, etc.)
    - otherwise Streamlit raises if a stored value falls outside a
      just-narrowed min/max.
  - Every profile other than Custom keeps using the global
    `FRONT_SPRING_MIN`/etc. constants unchanged - `effective_front_spring_min`
    and friends (set right after the profile selectbox) are the actual
    values used everywhere downstream, and only Custom overrides them.
- **Per-term enable/disable checkboxes.** Each objective term and each
  penalty *weight* (`underspring_penalty`, `damping_ratio_penalty` - not the
  target/tolerance/floor parameters those penalties use, which aren't
  independently-weighted categories) now has a checkbox next to its weight
  input. Unchecking one sends `0.0` for that key in the weights dict passed
  to `simulate_bump()` regardless of the number still shown in the input
  (the input is also visually disabled via `disabled=not term_enabled`) -
  so a term can be flipped off and back on without losing whatever weight
  was typed in.

### How this was verified

No pytest suite exists yet (see "Open items"). Used Streamlit's
`AppTest` harness (`streamlit.testing.v1.AppTest`) this time instead of the
constants-only extraction, since the change is UI-driven:

- Selected Custom, tightened the front spring range to 50,000-80,000, and
  confirmed the front spring slider's live min/max updated to match and its
  stored value got clamped into the new range.
- Set an inverted range (min > max) and confirmed a warning appeared and
  the slider fell back to the original 10,000-250,000 default rather than
  erroring.
- Unchecked a term's checkbox and confirmed no exception and the score
  still computed.
- Ran the full "Run Bayesian Optimization Loop" end-to-end (with
  `BO_NUM_INITIAL_SAMPLES`/`BO_TOTAL_ITERATIONS` temporarily lowered in a
  throwaway copy, purely to keep the smoke test fast) with a narrowed
  Custom range and a disabled term active, and confirmed it completed and
  reported an optimal setup with no exceptions.

## Earlier work: fixing a runaway realism penalty that was drowning out ride quality

**Symptom reported by the user:** near-identical parameter sets produced
wildly different scores (e.g. only changing front damping from 50,000 to
100 Ns/m swung the Comfort-profile cost from ~324 down to ~16), and a setup
that visibly never stopped oscillating through the whole 5-second sim
window could still score lower (better) than setups that clearly settled
down faster.

**Root cause, found by extracting the constants + `HalfCarPitchSimulator`
into a standalone script (same technique as the previous handoff) and
printing the per-term cost breakdown for the reported parameter sets:**
`compute_realism_penalties`'s damping-ratio term is `(tolerance_widths_past
_band) ** 2`, and `tolerance_widths_past_band` is unbounded above - an
overdamped setup (large damping relative to a soft spring's small critical
damping) can push it into the double digits before squaring. Every other
cost term is normalized to roughly 0-5; this one had no ceiling and could
reach the hundreds, silently making the score almost entirely a measure of
"how far is the damping ratio from my assumed target" rather than anything
about the simulated ride response. That explains both symptoms: a small
damping change that crosses from "somewhat under target" (bounded penalty,
since damping ratio can't go below 0) to "way over target" (unbounded
penalty) swings the score by 10-20x, and a persistently-oscillating but
*underdamped* setup was never penalized as heavily as an overdamped one,
regardless of how bad it actually looked in the response plot.

A secondary, related gap: none of the existing terms directly measure
"does the ride ever settle down." `post_heave_rms`/`post_heave_peak_to_peak`
average over the *whole* post-bump window, so a setup that decays sharply
in the first second and a setup that rings at low amplitude for the entire
window can land on similar RMS values despite being very different rides
to actually be in.

**What changed (all in `MAIN.py`):**

- **`DAMPING_RATIO_PENALTY_MAX_WIDTHS = 2.5`** (new constant, near the other
  `REF_*` constants) - the tolerance-widths-past-band value is now clipped
  to this before squaring, in `compute_realism_penalties`. This keeps the
  penalty strong enough to steer the optimizer away from genuinely
  degenerate damping choices without letting it dwarf every measured
  ride-quality term by orders of magnitude. (`underspring_penalty` didn't
  need an equivalent change - it's already naturally bounded to `[0, 1]`
  per wheel since the shortfall fraction saturates at zero spring rate.)
- **New term: `post_bump_persistence_ratio`.** Computed in `simulate_bump`
  by splitting the post-bump window in half and comparing late-window heave
  RMS to early-window heave RMS. A ratio near (or above) 1 means the car is
  shaking about as much at the end of the window as right after the bump -
  i.e. it isn't settling down - independent of whether the absolute
  amplitude is large. It's dimensionless by construction, so it needs no
  `REF_*` scaling constant, and it's added to the cost under the
  `oscillation_persistence` weight key (present in all three presets:
  2.5 for Comfort, 1.0 for Track, 1.5 for Balanced) and to
  `CUSTOM_PROFILE_OBJECTIVE_FIELDS` for the Custom profile UI.

### How this was verified

Same ad hoc approach as before (standalone extraction, no pytest suite
exists yet - see "Open items" below):

- Re-ran the exact parameter sets from the bug report. Comfort-profile
  costs went from `{10.29, 324.46, 16.13}` to `{12.36, 30.99, 17.41}` for
  the same three setups (mild damping / overdamped front / underdamped
  front+rear respectively) - the overdamped case is still correctly scored
  worst (extreme overdamping is a real harshness problem), but by a factor
  of ~2.5x instead of ~20-30x, and a genuinely well-damped comparison setup
  (both ends at a moderate damping ratio) now clearly scores best (6.74) of
  the four, which wasn't true before the fix.
- Re-checked the three-preset sanity sweep from the original cost-function
  pass (soft/undamped vs. mid vs. stiff/well-damped setups) - Comfort and
  Balanced still favor the mid-to-stiff, well-damped setup; Track still
  favors the stiffer setup over the soft one; ordering is unchanged, only
  the absolute scale changed.
- Re-confirmed passing an explicit dict equal to a preset's dict still
  produces a bit-identical cost to calling with that preset's name.

If you touch the cost function again, rebuild this kind of throwaway check
(or, better, finally add the `pytest` setup mentioned below) before
trusting the optimizer's output.

## Earlier work: cost function realism pass

The original cost function combined several raw physical quantities (peak
pitch acceleration, heave RMS, etc.) with hand-tuned weights in the
thousands, and the "Track Focused" profile had **no realism penalties at
all** — `weights.get("damping_ratio_penalty", 0.0)` was silently returning
0 because the key didn't exist in that profile's dict, so the optimizer
could hand back a track setup with near-zero damping and nothing would stop
it.

What changed (all in `MAIN.py`, roughly lines 193–330 for constants/weights,
and inside `HalfCarPitchSimulator.simulate_bump` /
`compute_realism_penalties` for the math):

- **Normalization.** Every raw cost term is now divided by a `REF_*`
  constant (documented inline, near the top of the file) before being
  weighted, so all profile weights live on a common ~0–5 "relative
  importance" scale instead of being opaque four/five-digit numbers tuned
  to one specific bump height and vehicle mass.
- **New term: `bump_pitch_accel_rms`.** RMS pitch acceleration across the
  bump event, alongside the existing peak. A peak alone misses sustained
  shaking — this is a lightweight nod to ISO 2631's frequency-weighted RMS
  approach without implementing a full weighting filter.
- **New term: `tire_load_variation`.** Dynamic tire load swing during the
  bump (computed from the same tire force equations already used in the
  ODE, evaluated post-hoc from the solved wheel trajectories). This is what
  actually matters for grip/contact-patch consistency, and it's now the
  dominant term in the Track profile — previously Track only had a mild
  `spring_rate_sum` term, which doesn't mean the same thing.
- **Per-profile realism targets.** `damping_ratio_target`,
  `damping_ratio_tolerance`, and `spring_rate_min` moved from shared global
  constants (`COMFORT_DAMPING_RATIO_TARGET`, etc. — now removed) into each
  profile's own dict. Track now has a real damping-ratio target (0.65) and
  spring floor (40,000 N/m) distinct from Comfort's (0.35 / 25,000 N/m).
- **Normalized penalty math.** Underspring penalty is now a fractional
  shortfall relative to the profile's own floor (dimensionless, ~0–1)
  instead of a raw squared N/m number that needed a `2e-7`-scale
  compensating weight. Damping-ratio penalty is expressed in "tolerance
  widths past the allowed band" for the same reason.
- **`simulate_bump(..., profile=...)` now accepts either a preset name
  (string, looked up in `PROFILE_WEIGHTS`) or a fully-formed weights dict.**
  This is what makes the custom profile work — see below — and it's a
  backward-compatible change (existing string calls still work).
- **Custom profile UI.** The sidebar profile selectbox gained a "Custom"
  option. Selecting it opens an expander with a `st.number_input` for every
  objective term and every realism parameter, defaulting to the Balanced
  preset's values (`CUSTOM_PROFILE_DEFAULTS`). The field metadata
  (label/min/max/step/help text) lives in `CUSTOM_PROFILE_OBJECTIVE_FIELDS`
  and `CUSTOM_PROFILE_REALISM_FIELDS` near the top of the file — add a
  tuple there if a new cost term needs a UI control later.
  `selected_profile_name` (string, for display) and `selected_profile`
  (string or dict, passed to `simulate_bump`) are both in scope after this
  block; **use `selected_profile_name` in any new user-facing labels**, not
  `selected_profile`, or a custom run will print a dict where a profile name
  is expected.

### How this was verified

There's no formal test suite in this repo. Verification for this change was
done ad hoc: the constants + `HalfCarPitchSimulator` class were extracted
into a standalone script (no Streamlit dependency, since that section of
`MAIN.py` has none) and exercised with a handful of representative
spring/damping combinations across all three presets, checking that:

- Comfort scores a mid-range setup better than either extreme.
- Track now scores an under-damped/soft setup clearly worse than a
  stiff/well-damped one (it couldn't distinguish these before).
- Passing an explicit dict equal to a preset's dict produces a bit-identical
  cost to calling with that preset's name.

If you change the cost function again, recreate a similar throwaway check
before trusting the optimizer's output — there's currently no `pytest`
setup, `conftest.py`, or CI. Adding one would be a reasonable next step if
this project keeps growing.

## Open items / suggested next steps

- **Track/cone AI-driving mode (this handoff's most recent work) has never
  been run against a live BeamNG instance.** See "What's unverified" above
  for the specific list — confirming `BEAMNG_CONE_PROP_MODEL`, checking
  whether cones actually spawn on this build the same way the vehicle
  needed a live-spawn workaround, and confirming `Timer`/`GForces`/`Damage`
  field names are the concrete next steps, roughly in that order.
- **`DAMPING_RATIO_PENALTY_MAX_WIDTHS = 2.5` is a hand-picked ceiling, not
  derived from anything.** It was chosen to keep the realism penalty's
  worst case in the same rough order of magnitude as the sum of the other
  weighted terms, not from a formal analysis. If profiles ever get pushed
  to search wildly different damping/spring ranges, sanity-check this
  again with the same standalone-extraction technique.
- **`post_bump_persistence_ratio`'s early/late split is a simple 50/50
  time-window heuristic, not a real decay-envelope or logarithmic-decrement
  fit.** It catches "still shaking about as much at the end as at the
  start" but won't distinguish more nuanced decay shapes (e.g. a setup that
  decays quickly then picks back up due to a secondary resonance). A proper
  fit to an exponential envelope would be more rigorous if this becomes
  important.
- **`REF_*` normalization constants are hand-picked, not derived.** They're
  documented inline as calibration constants representing "typical"
  response magnitudes, but nobody has run a systematic sweep to confirm
  they're centered correctly. If cost values start looking skewed (e.g.
  one term always dominates regardless of profile), check these first.
- **No automated tests.** See above — worth adding if this project is
  going to keep evolving under multiple contributors.
- **BeamNG bridge is version-pinned by observation, not by design.** The
  compatibility shim in the `beamngpy` import block (`_patched_get_current_info`)
  exists because of a specific BeamNG.drive build (0.39.4.0.20972) /
  `beamngpy` 1.36 mismatch. If either is upgraded, re-check whether the shim
  is still needed or now actively wrong.
- **No frequency-weighted (true ISO 2631) comfort metric.** `bump_pitch_accel_rms`
  is a cheap approximation. A full Wk-filter implementation would be more
  rigorous if ride-comfort accuracy becomes a priority.
- **Roll dynamics are out of scope by design** (half-car / pitch-plane
  only) — not a bug, just a documented modeling limitation in `README.md`.

## File map

- `MAIN.py` — everything: constants, physics model, BeamNG bridge (both the
  suspension-bump validation and the newer track/cone AI-driving mode),
  Bayesian optimization loop, Streamlit UI. Single file, ~2,760 lines as of
  this handoff.
- `README.md` — user/developer-facing project description, kept in sync
  with actual code behavior (not aspirational marketing copy). **Not yet
  updated for the track/cone feature** — do that alongside confirming the
  open items above, so it doesn't get written from aspiration instead of
  verified behavior (see this file's own project-description note above).
- `TRACK_CONE_AI_INTEGRATION.md` — design spec for the track/cone AI-driving
  feature, written before implementation. `MAIN.py`'s new `1.5`/`2.5`
  sections are a direct implementation of it; read them together, and treat
  this doc's §9 ("Open items to flag") as still current.
- `HANDOFF.md` — this file.
