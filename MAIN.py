"""
AI-Powered Suspension Tuning Advisor
=====================================

This app models a car's suspension as a "half-car pitch model": a single 2D
slice of the vehicle seen from the side, capturing bounce (up/down) and
pitch (nose up/down rotation) but ignoring roll (side to side tipping).

Pipeline:
    1. HalfCarPitchSimulator integrates the equations of motion (Newton's
       2nd law for each mass, F = m*a) for the car driving over a bump,
       and reports a scalar "cost" (peak pitch acceleration) representing
       how harsh/uncomfortable the ride is.
    2. A Gaussian Process (GP) Bayesian Optimization loop treats the
       simulator as a black-box function cost = f(spring/damper settings)
       and searches for the settings that minimize that cost, using far
       fewer simulator evaluations than a brute-force grid search.
    3. Optionally, the chosen settings can be validated in BeamNG.drive
       (a full 3D physics game) to sanity-check the simplified math model
       against a much higher-fidelity simulation.

Everything is wired up in a Streamlit dashboard for interactive tuning.

How this file is organized (read top to bottom):
    The file is split into the five numbered sections marked with "====="
    banners below: (1) the physics simulator, (1.5) track/cone geometry,
    (2) the BeamNG bridge for the suspension-bump test, (2.5) the BeamNG
    bridge for the track/cone AI-driving test, and (3) the Streamlit UI
    script that ties everything together and is what actually runs when
    you launch the app.

    Within each section, functions/methods are ordered so that a function
    appears ABOVE the helper functions it calls - e.g. simulate_bump()
    (the main entry point of the simulator) appears right after __init__,
    with the lower-level math it depends on (compute_static_equilibrium,
    calculate_system_equations, compute_realism_penalties, ...) below it.
    The idea is that you can read a section start-to-finish, top to bottom,
    the same way you'd trace a call stack, instead of having to jump
    around the file to find out what a function actually does.
    One exception: TrackPoint/TrackDefinition (section 1.5) are simple
    data classes referenced by type hints throughout that section, so -
    like a struct definition - they're placed before everything that uses
    them rather than being ordered by who calls whom.
    Section 3 (the Streamlit script) is different: it's a linear script
    that builds the UI top-to-bottom and relies on variables/widgets set
    up earlier in the script, so its statements stay in their original,
    execution order rather than being reshuffled by call graph.
"""

import os
import csv
import io
import json
import math
import random
import time
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from scipy.integrate import solve_ivp   # numerical ODE integrator (Runge-Kutta family)
from scipy.optimize import minimize, root  # `root` solves the nonlinear static-equilibrium equations;
                                            # `minimize` finds the best next point to sample during
                                            # Bayesian Optimization (minimizing the acquisition function).
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Try importing beamngpy (handled gracefully if the game isn't running locally)
try:
    import beamngpy
    from beamngpy import BeamNGpy, Scenario, Vehicle, angle_to_quat
    from beamngpy.sensors import Electrics
    # NOTE: AdvancedIMU is intentionally not imported/used. It is a
    # BeamNG.tech-only sensor and is rejected server-side with
    # "This feature requires a BeamNG.tech license" on plain BeamNG.drive.
    # Vertical acceleration is instead derived from finite-differencing the
    # free `state` sensor's velocity - see run_beamng_test(). Electrics
    # IS free/non-tech and exposes wheel-speed data, so it's attached below
    # to pull in real per-wheel telemetry where available.
    BEAMNG_AVAILABLE = True

    # --- Compatibility shim -------------------------------------------------
    # This BeamNG.drive build (0.39.4.0.20972) responds to the internal
    # "GetCurrentVehicles" request with a JSON *list* of vehicle entries,
    # while beamngpy==1.36's VehicleApi.get_current_info() still assumes the
    # older dict-of-{vehicle_id: vehicle_data} shape and calls info.items()
    # directly, raising AttributeError. beamngpy tracks BeamNG.tech more
    # closely than BeamNG.drive, so this kind of drift shows up on newer
    # .drive builds before an official beamngpy release catches up.
    # Normalize the response shape at the source so every caller downstream
    # (get_current, _load_existing_vehicles, etc.) keeps working unmodified.
    from beamngpy.api.beamng.vehicles import VehiclesApi as _BngVehicleApi

    _original_get_current_info = _BngVehicleApi.get_current_info

    def _patched_get_current_info(self, include_config: bool = True):
        info = self._message("GetCurrentVehicles", include_config=include_config)
        if not info:
            return {}
        if isinstance(info, list):
            # Normalize list-of-vehicle-dicts -> {vehicle_id: vehicle_dict}
            normalized = {}
            for entry in info:
                if not isinstance(entry, dict):
                    continue
                vid = entry.get("id") or entry.get("vid") or entry.get("name")
                if vid is None:
                    continue
                normalized[str(vid)] = entry
            info = normalized
        for vid, vehicle in info.items():
            if "id" in vehicle:
                vehicle["id"] = int(vehicle["id"])
        return info

    _BngVehicleApi.get_current_info = _patched_get_current_info
    # -------------------------------------------------------------------------

except ImportError:
    beamngpy = None
    BEAMNG_AVAILABLE = False

# ==========================================
# DEBUG PROTOCOL / VERSION INSPECTION
# ==========================================
import sys
st.sidebar.markdown("### 🛠️ Environment Diagnostics")
st.sidebar.code(f"Python: {sys.executable}\nBeamNGpy: {beamngpy.__version__ if BEAMNG_AVAILABLE else 'Not Loaded'}")

# ==========================================
# BEAMNG DIRECTORY DETECTION & VALIDATION
# ==========================================
def validate_beamng_directory(path_str: str) -> bool:
    """Check if the provided path is a valid BeamNG home directory containing game executables."""
    if not path_str:
        return False
    
    path = Path(path_str)
    if not path.exists() or not path.is_dir():
        return False

    # Check for expected game binaries/executables
    valid_executables = [
        path / "BeamNG.drive.x64.exe",
        path / "Bin64" / "BeamNG.drive.x64.exe",
        path / "Bin64" / "BeamNG.drive.v0.31.x64.exe"
    ]
    return any(exe.exists() for exe in valid_executables)


def detect_beamng_home() -> str | None:
    """Automatically search standard paths and environment variables for BeamNG.drive."""
    # 1. Check environment variable
    if "BNG_HOME" in os.environ and validate_beamng_directory(os.environ["BNG_HOME"]):
        return os.environ["BNG_HOME"]

    # 2. Check standard Steam library paths across common drives
    standard_paths = [
        r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive",
        r"C:\Program Files\Steam\steamapps\common\BeamNG.drive",
        r"D:\SteamLibrary\steamapps\common\BeamNG.drive",
        r"E:\SteamLibrary\steamapps\common\BeamNG.drive",
        r"F:\SteamLibrary\steamapps\common\BeamNG.drive",
    ]

    for candidate in standard_paths:
        if validate_beamng_directory(candidate):
            return candidate

    return None


def select_folder_gui() -> str | None:
    """Launch a native desktop folder picker dialog to select the folder manually."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        selected_folder = filedialog.askdirectory(title="Select BeamNG.drive Main Directory")
        root.destroy()
        return selected_folder if selected_folder else None
    except Exception:
        return None


# ==========================================
# GLOBAL CONFIGURATION & TUNING CONSTANTS
# ==========================================

# --- Suspension Parameter Bounds & Step Sizes ---
FRONT_SPRING_MIN, FRONT_SPRING_MAX, FRONT_SPRING_STEP = 10000, 250000, 1000
FRONT_DAMPING_MIN, FRONT_DAMPING_MAX, FRONT_DAMPING_STEP = 100, 50000, 100
REAR_SPRING_MIN, REAR_SPRING_MAX, REAR_SPRING_STEP = 10000, 250000, 1000
REAR_DAMPING_MIN, REAR_DAMPING_MAX, REAR_DAMPING_STEP = 100, 50000, 100

SEARCH_BOUNDS = [
    (FRONT_SPRING_MIN, FRONT_SPRING_MAX),
    (FRONT_DAMPING_MIN, FRONT_DAMPING_MAX),
    (REAR_SPRING_MIN, REAR_SPRING_MAX),
    (REAR_DAMPING_MIN, REAR_DAMPING_MAX)
]

# --- Custom-profile search-range guardrails ---
# The Custom profile (sidebar) lets the user override the spring/damping
# search bounds above for both the manual sliders and the Bayesian
# Optimization loop. These absolute limits keep those overrides inside a
# range where the physics model still behaves: a spring rate of 0 would
# divide by zero in compute_damping_ratios(), and pushing either value to
# extremes can make solve_ivp's adaptive step size collapse.
CUSTOM_SPRING_RANGE_ABSOLUTE_LIMITS = (1000, 500000)   # N/m
CUSTOM_DAMPING_RANGE_ABSOLUTE_LIMITS = (0, 100000)     # Ns/m

# --- Vehicle Layout & Fundamental Physical Constants ---
# BEAMNG ETK800 AS REFERENCE
BEAMNG_AI_SPEED = 15.0         
CHASSIS_SPRUNG_MASS = 1455.0   
FRONT_UNSPRUNG_MASS = 40.0     
REAR_UNSPRUNG_MASS = 40.0      
DISTANCE_FRONT_AXLE = 1.3958    
DISTANCE_REAR_AXLE = 1.4498    
TIRE_STIFFNESS = 200000.0      
TIRE_DAMPING = 2500.0          
GRAVITY = 9.81                 

WHEELBASE = DISTANCE_FRONT_AXLE + DISTANCE_REAR_AXLE
REAR_AXLE_ROAD_LAG = WHEELBASE / BEAMNG_AI_SPEED 
PITCH_INERTIA = CHASSIS_SPRUNG_MASS * (DISTANCE_FRONT_AXLE * DISTANCE_REAR_AXLE) 

# --- Simulation Profile Settings ---
SIM_TIME_START = 0.0     
SIM_TIME_END = 20.0       
SIM_TIME_POINTS = 150    

BUMP_START_TIME = 1.0
BUMP_HEIGHT = 0.05
SINGLE_BUMP_LENGTH = 5.0     # Length of one individual bump (e.g., meters)
TOTAL_BUMPS = 5              # Number of sequential bumps

# Automatically derived track properties
TOTAL_BUMP_DISTANCE = SINGLE_BUMP_LENGTH * TOTAL_BUMPS
SINGLE_BUMP_DURATION = SINGLE_BUMP_LENGTH / BEAMNG_AI_SPEED
TOTAL_BUMP_DURATION = SINGLE_BUMP_DURATION * TOTAL_BUMPS

BUMP_END_TIME = BUMP_START_TIME + TOTAL_BUMP_DURATION

BUMP_EVENT_END = BUMP_END_TIME + REAR_AXLE_ROAD_LAG + 0.15
POST_BUMP_START = BUMP_END_TIME + REAR_AXLE_ROAD_LAG + 0.2

# --- Cost Function Normalization References ---
# Each raw response quantity below is measured in different units (rad/s^2,
# m, m/s, N, N/m) and spans wildly different natural magnitudes. Dividing
# each one by a reference value of the right order of magnitude turns it
# into a dimensionless, roughly-O(1) quantity *before* it is weighted. That
# means profile weights below represent "relative importance" on a common
# 0-5-ish scale instead of opaque four/five-digit numbers that only made
# sense for one specific bump height and vehicle mass. If BUMP_HEIGHT,
# CHASSIS_SPRUNG_MASS, or similar constants change significantly, revisit
# these references (they should sit near the typical value produced by a
# "reasonable" mid-range suspension setup under the current scenario).
REF_PEAK_PITCH_ACCEL = 3.0          # rad/s^2 - typical peak pitch accel over the bump
REF_BUMP_PITCH_ACCEL_RMS = 1.2      # rad/s^2 - typical RMS pitch accel across the bump event
REF_POST_HEAVE_RMS = 0.004          # m - typical post-bump settling heave RMS
REF_POST_PITCH_VEL_RMS = 0.15       # rad/s - typical post-bump pitch velocity RMS
REF_POST_HEAVE_ACCEL_RMS = 1.0      # m/s^2 - typical post-bump heave accel RMS
REF_POST_HEAVE_PEAK_TO_PEAK = 0.01  # m - typical post-bump heave peak-to-peak
REF_TIRE_LOAD_VARIATION = 1500.0    # N - typical dynamic tire load swing during the bump
REF_SPRING_RATE_SUM = 3.0e5         # N/m - front + rear spring rate at a mid-range setup

# post_bump_persistence_ratio (late-window heave RMS / early-window heave RMS,
# see compute below) is already a dimensionless ~0-1ish ratio by construction,
# so it needs no REF_* scaling constant of its own.

# --- Realism Penalty Ceiling ---
# compute_realism_penalties() expresses damping-ratio error in units of
# "tolerance widths past the allowed band". That quantity is unbounded above
# (damping can be dialed arbitrarily high relative to a soft spring's small
# critical damping), so squaring it without a ceiling let a single
# borderline-unrealistic damping choice produce a penalty in the hundreds -
# two to three orders of magnitude past every other term's ~0-5 range - and
# silently take over the entire cost score. Capping the "widths past band"
# value before squaring keeps the penalty able to strongly discourage
# degenerate setups without being able to drown out the actual measured ride
# response for merely suboptimal (but not degenerate) ones.
DAMPING_RATIO_PENALTY_MAX_WIDTHS = 2.5

# --- Ride Comfort Note (ISO 2631-inspired) ---
# Raw peak pitch acceleration alone is a crude harshness proxy: it reflects
# only a single instant and ignores that humans are far more sensitive to
# vibration in some frequency bands than others (the ISO 2631 whole-body
# vibration standard weights the roughly 4-8 Hz vertical band most heavily).
# A full frequency-weighting filter is out of scope for this half-car model,
# but adding an RMS-over-the-bump-event term (bump_pitch_accel_rms) alongside
# the peak captures sustained harshness that a single peak can miss - two
# setups can share the same peak value but differ a lot in how long the
# shaking after it lasts.

# --- Profile Presets ---
# Each preset fully specifies both the objective weights (relative
# importance, applied to the normalized/dimensionless terms above) and the
# realism-penalty targets (damping ratio target/tolerance, minimum spring
# rate) used by compute_realism_penalties(). Every profile - including
# Track, which previously had none - carries nonzero realism penalties so
# the optimizer can't converge on a physically degenerate setup (near-zero
# damping, or springs so soft the tire nearly free-falls) just because that
# minimizes the raw objective.
PROFILE_WEIGHTS = {
    "Comfort (Smooth Ride)": {
        # Objective terms (weights on normalized, dimensionless quantities)
        "peak_pitch_accel": 2.5,
        "bump_pitch_accel_rms": 2.0,
        "post_heave_rms": 2.0,
        "post_pitch_vel_rms": 1.2,
        "post_heave_accel_rms": 1.0,
        "post_heave_peak_to_peak": 1.0,
        "oscillation_persistence": 2.5,  # a ride that keeps bouncing is the definition of "not comfortable"
        "tire_load_variation": 0.3,   # grip consistency is not the priority here
        "spring_rate_sum": 0.0,
        # Realism penalties - keep a soft, well-damped setup physically plausible
        "underspring_penalty": 2.0,
        "damping_ratio_penalty": 3.0,
        "damping_ratio_target": 0.35,
        "damping_ratio_tolerance": 0.15,
        "spring_rate_min": 25000.0,
    },
    "Track Focused (Stiff & Responsive)": {
        "peak_pitch_accel": 0.6,
        "bump_pitch_accel_rms": 0.4,
        "post_heave_rms": 1.0,
        "post_pitch_vel_rms": 1.0,
        "post_heave_accel_rms": 0.5,
        "post_heave_peak_to_peak": 0.5,
        "oscillation_persistence": 1.0,  # still relevant for body control, just not the priority
        "tire_load_variation": 3.0,   # dominant term: keep the contact patch loaded consistently
        "spring_rate_sum": 0.3,
        # Realism penalties - higher damping-ratio target for body control,
        # plus a nonzero underspring penalty and spring floor so the search
        # can't null out the springs (previously unconstrained).
        "underspring_penalty": 1.0,
        "damping_ratio_penalty": 2.5,
        "damping_ratio_target": 0.65,
        "damping_ratio_tolerance": 0.15,
        "spring_rate_min": 40000.0,
    },
    "Balanced": {
        "peak_pitch_accel": 1.5,
        "bump_pitch_accel_rms": 1.2,
        "post_heave_rms": 1.5,
        "post_pitch_vel_rms": 1.2,
        "post_heave_accel_rms": 0.8,
        "post_heave_peak_to_peak": 0.8,
        "oscillation_persistence": 1.5,
        "tire_load_variation": 1.0,
        "spring_rate_sum": 0.05,
        "underspring_penalty": 1.5,
        "damping_ratio_penalty": 2.0,
        "damping_ratio_target": 0.50,
        "damping_ratio_tolerance": 0.18,
        "spring_rate_min": 30000.0,
    },
}

# Default values a new "Custom" profile starts from (the Balanced preset),
# and the field metadata the sidebar uses to build number inputs generically:
# (key, label, min, max, step, help_text).
CUSTOM_PROFILE_DEFAULTS = dict(PROFILE_WEIGHTS["Balanced"])
CUSTOM_PROFILE_OBJECTIVE_FIELDS = [
    ("peak_pitch_accel", "Peak pitch acceleration", 0.0, 5.0, 0.1,
     "Weight on the single sharpest pitch-acceleration spike during the bump."),
    ("bump_pitch_accel_rms", "Bump pitch accel. RMS", 0.0, 5.0, 0.1,
     "Weight on sustained (RMS) pitch acceleration across the bump event, not just the peak."),
    ("post_heave_rms", "Post-bump heave RMS", 0.0, 5.0, 0.1,
     "Weight on how much the chassis keeps bouncing after the bump."),
    ("post_pitch_vel_rms", "Post-bump pitch velocity RMS", 0.0, 5.0, 0.1,
     "Weight on how much the chassis keeps rocking (nose up/down) after the bump."),
    ("post_heave_accel_rms", "Post-bump heave accel. RMS", 0.0, 5.0, 0.1,
     "Weight on jerkiness of the settling motion after the bump."),
    ("post_heave_peak_to_peak", "Post-bump heave peak-to-peak", 0.0, 5.0, 0.1,
     "Weight on the overall size of the rebound/overshoot after the bump."),
    ("oscillation_persistence", "Oscillation persistence", 0.0, 5.0, 0.1,
     "Weight on whether the ride keeps bouncing instead of settling down - compares heave "
     "in the back half of the post-bump window to the front half, so a setup that never "
     "damps out is penalized even if its RMS motion isn't huge in absolute terms."),
    ("tire_load_variation", "Dynamic tire load variation", 0.0, 5.0, 0.1,
     "Weight on how much tire load swings during the bump - lower swing means more consistent grip."),
    ("spring_rate_sum", "Overall spring stiffness", 0.0, 2.0, 0.05,
     "Mild regularizer discouraging (or, near zero, permitting) very stiff overall springing."),
]
CUSTOM_PROFILE_REALISM_FIELDS = [
    ("underspring_penalty", "Underspring penalty weight", 0.0, 5.0, 0.1,
     "Weight discouraging springs softer than the minimum spring rate below."),
    ("damping_ratio_penalty", "Damping ratio penalty weight", 0.0, 5.0, 0.1,
     "Weight discouraging a damping ratio far from the target below."),
    ("damping_ratio_target", "Target damping ratio", 0.1, 1.0, 0.01,
     "Desired damping ratio (fraction of critical damping). ~0.3-0.4 comfort, ~0.6-0.7 track."),
    ("damping_ratio_tolerance", "Damping ratio tolerance", 0.02, 0.5, 0.01,
     "How far the damping ratio may drift from the target before the penalty kicks in."),
    ("spring_rate_min", "Minimum spring rate (N/m)", 10000.0, 100000.0, 1000.0,
     "Floor below which a spring rate is considered unrealistically soft."),
]

# --- BeamNG Validation Settings ---
BEAMNG_HOST = 'localhost'
BEAMNG_PORT = 25252
BEAMNG_MAP = 'gridmap_v2'
BEAMNG_VEHICLE_MODEL = 'etkc'
BEAMNG_VEHICLE_LICENSE = 'AI_TEST'
BEAMNG_SPAWN_POS = (-3000, 1000, 100)
# Rotation captured from the World Editor Inspector (rotationEuler Z = 180 deg,
# X/Y = 0), originally converted straight to quaternion (x, y, z, w) as
# (0, 0, 1, 0) - a pure 180 deg yaw about Z. In practice that made the car
# spawn facing backward relative to the intended driving direction (the
# throttle-forward test drive was reversing down the road), so the World
# Editor's Z-rotation reading and beamngpy's rot_quat evidently don't agree
# on which way is "forward" here. Flipping by another 180 deg cancels back
# out to identity (no extra yaw), which is the fix below. If the car is
# still facing the wrong way after this, re-check the heading in the World
# Editor and adjust this quaternion again - a pure yaw of theta degrees
# about Z is (0, 0, sin(theta/2), cos(theta/2)).
BEAMNG_SPAWN_ROT_QUAT = (0, 0, 1, 0)
BEAMNG_IMU_FREQ = 50           
BEAMNG_TEST_DURATION = 180     
BEAMNG_POLL_INTERVAL = 0.02    

# Track files (see TRACK_CONE_AI_INTEGRATION.md §1) are authored in a small
# LOCAL frame centered near their own (0, 0) for easy editing/portability
# across levels - e.g. figure8_track.csv spans roughly -25..25 in x/y. That
# is NOT the same frame as BEAMNG_SPAWN_POS, despite TrackPoint's docstring
# assuming it already would be. Taken literally, every uploaded track was
# being built right on top of gridmap_v2's own world origin, which is the
# map's crowded central hub - which is why the track flow always looked
# like it was "spawning in the default location" no matter what the
# scenario/spawn code did. load_track_file() now translates every point by
# this world-space anchor (x, y only - z is left alone, since track files
# already give a real ground height like the sample's z=100) so a track
# authored around its own (0,0) lands wherever this anchor points to
# instead. Defaults to BEAMNG_SPAWN_POS's x/y since that's already been
# confirmed as clear ground; override via the "Track World Origin" inputs
# in the Track tab to place a track elsewhere on the map.
TRACK_WORLD_ORIGIN_XY = (BEAMNG_SPAWN_POS[0], BEAMNG_SPAWN_POS[1])

# The track flow used to spawn the car at the track's first point using the
# same FIXED BEAMNG_SPAWN_ROT_QUAT as the bump-test flow, regardless of
# which direction the track actually heads off in. Since that rotation was
# only ever validated for the bump test's own straight-line heading, on
# most tracks the car would spawn facing the wrong way relative to the
# path - and BeamNG's AI script driver responds to that by backing/turning
# in place to correct its heading before actually driving the path, which
# is the "wants to drive in reverse to align itself" behavior. Two things
# address this:
#   1. The car is now spawned RUN_UP_DISTANCE_M behind the track's first
#      point, facing along the initial tangent, with a matching run-up
#      node prepended to the AI script - giving it room to roll up to
#      speed and settle onto the path instead of starting from a dead
#      stop exactly on top of node 0.
#   2. The spawn rotation is now derived from the track's own initial
#      heading instead of the fixed constant.
# The heading->quat conversion below WAS empirically verified in-game: at
# the +180 deg offset it still drove backward at the start of every track
# run (confirmed by the user), so this was flipped to +0.0 deg (i.e.
# heading_to_spawn_quat() just uses the initial tangent's heading
# directly, with no correction). That fixed the forward/backward (180 deg)
# problem, but the user then reported a separate 90 deg error: at +0.0 the
# car spawns rotated 90 deg counterclockwise from where it should be -
# parallel to the start/finish gate instead of perpendicular to it (see
# the RUN_UP_DISTANCE_M note just below for why perpendicular is correct).
# Countering a 90-deg-CCW error means rotating the heading 90 deg
# clockwise, i.e. subtracting 90 deg, so this is now -90.0. If that
# overshoots and the car instead spawns rotated 90 deg clockwise (i.e. the
# error flips sign instead of disappearing), the fix is the other
# direction, +90.0 - see BEAMNG_SPAWN_ROT_QUAT's comment above for why a
# load-bearing heading constant like this needs to be re-verified in-game
# rather than assumed.
#
# RUN_UP_DISTANCE_M is the run-up gap ahead of the start/finish gate the
# car is spawned at, facing straight along the track's initial tangent -
# which is perpendicular to the gate itself (see compute_finish_line() /
# compute_run_up_start()), so the car meets the line squared-up rather
# than at an angle. Set to 5.0 m per the user's requested spawn gap.
RUN_UP_DISTANCE_M = 25.0
TRACK_HEADING_ROT_OFFSET_DEG = 75.0

# --- Start/finish line + run-timeout settings (see compute_finish_line()
# and run_beamng_track_test() below) ---
# Hard safety-net timeout for a track run, independent of the AI script's
# own scheduled duration. A run is meant to always end on its own via the
# start/finish line crossing; this constant only matters if the car
# crashes, gets stuck, or overcorrects badly enough that it never reaches
# the finish - without it, a bad run just hangs the UI in "running"
# indefinitely.
ABSOLUTE_RUN_FALLBACK_S = 300.0
# The "normal" (non-fallback) timeout is scaled off the AI script's own
# scripted duration, but generously - a car that's fighting to stay on
# line (overcorrecting) can take noticeably longer than the script's
# idealized constant-speed timing assumed. run_beamng_track_test() takes
# the larger of this and ABSOLUTE_RUN_FALLBACK_S, so short tracks still
# get the full fallback window and long tracks get proportionally more.
RUN_TIMEOUT_SCRIPT_MULTIPLIER = 2.5
RUN_TIMEOUT_SCRIPT_PADDING_S = 30.0

# vehicles.spawn() defaults cling=True, which snaps z to the actual ground
# height at spawn time - but vehicles.teleport() (used right after, to force
# the position - see load_beamng_scenario()) has no such option and places
# the vehicle at the literal z given. A hardcoded/CSV-supplied z that's off
# by even a few centimeters from the real terrain there means the teleport
# embeds the wheels slightly instead of snapping clear, which can look like
# the car being "stuck." This small upward margin is added only to
# teleport()'s z (never spawn()'s, which doesn't need it) as cheap
# insurance - the vehicle settles the rest of the way under gravity.
TELEPORT_Z_CLEARANCE_M = 0.15

# --- Track / Cone AI-Driving Settings ---
# See TRACK_CONE_AI_INTEGRATION.md for the full design rationale. This is a
# separate scenario/flow from the suspension-bump validation above - it
# reuses the same beamngpy bridge patterns (empty scenario + live spawn,
# free-license-only sensors) but drives a track loaded from a CSV/JSON file
# instead of a single fixed bump.
#
# Confirmed in-game by the user: the real part-config path for the small
# orange/white traffic cone on this BeamNG install. This used to be left as
# the placeholder "small", which wasn't a real path at all - but the path
# format itself ("vehicles/cones/...") is also the reason cones are spawned
# as their own Vehicle instances below (load_beamng_track_scenario()),
# not as scenario StaticObjects like an earlier version of this file tried:
# "vehicles/<model>/<part>.pc" is BeamNG's layout for a vehicle model's part
# config, not a static-mesh shape path, so cones in this build's asset
# layout are pseudo-vehicles (spawnable, physical, jbeam-backed) rather than
# fixed decoration - which also means BEAMNG_CONE_VEHICLE_MODEL right below
# has to match ("cones", the folder this part config lives under).
BEAMNG_CONE_PROP_MODEL = "vehicles/cones/small_orange_white_single.pc"
BEAMNG_CONE_VEHICLE_MODEL = "cones"
DEFAULT_TRACK_WIDTH_M = 10.0
DEFAULT_TRACK_SPEED_MPS = BEAMNG_AI_SPEED
DEFAULT_CONE_SPACING_M = 5.0
MIN_TRACK_POINT_SPACING_M = 0.5
MIN_SCRIPT_DT = 0.05  # seconds; floor for execute_script's per-node timing,
                       # see build_ai_script() below
TRACK_AI_SCRIPT_RESAMPLE_SPACING_M = 4.0  # centerline resampling interval for
                                           # the AI script - deliberately
                                           # coarser than cone spacing (see
                                           # TRACK_CONE_AI_INTEGRATION.md §4)
TRACK_CONE_HIT_DAMAGE_DELTA_THRESHOLD = 0.01  # heuristic; see run_beamng_track_test()
TRACK_CONE_HIT_DISTANCE_THRESHOLD_M = 2.5     # heuristic; see run_beamng_track_test()
# The start/finish "line" (see compute_finish_line()) is really a finite
# gate, not an infinite line - only a lateral offset (perpendicular to the
# tangent) within this many meters of the start/finish point counts as
# actually crossing the gate. Without this bound, a closed-loop track
# whose own path runs parallel to the gate's direction anywhere on the map
# (e.g. a rectangular track with an edge perpendicular to the start
# straight) could be misread as crossing the gate far from the actual
# start/finish. Generous relative to DEFAULT_TRACK_WIDTH_M to tolerate a
# car overcorrecting off the racing line near the gate, but still bounded.
FINISH_LINE_HALF_WIDTH_M = max(DEFAULT_TRACK_WIDTH_M, 15.0)

# For a closed loop, the start line and finish line are the same physical
# gate (see compute_finish_line()) - "cross it once to start the lap,
# cross it again to finish." But the car can legitimately cross that same
# gate more than twice near the very start of a run without ever having
# driven the lap: the run-up alignment (see RUN_UP_DISTANCE_M/
# TRACK_HEADING_ROT_OFFSET_DEG above) can wobble it back and forth over
# the line, and each forward crossing looks identical to a real finish to
# the plain sign-flip check. Requiring the car to have also passed a
# handful of checkpoints spread around the rest of the track before a gate
# crossing is accepted as the finish rules that out, since a checkpoint on
# the far side of the track can't be reached by wobbling near the start.
# See select_finish_checkpoints() / run_beamng_track_test().
FINISH_CHECKPOINT_COUNT = 5
# How close the car has to get to a checkpoint to count as having passed
# it. Generous relative to cone spacing (checkpoints ARE cones - see
# select_finish_checkpoints()) so normal cross-track wander near a
# checkpoint still counts as passing it, but well inside
# FINISH_CHECKPOINT_COUNT's arc-length spacing on any track long enough to
# be worth gating in the first place.
FINISH_CHECKPOINT_RADIUS_M = max(DEFAULT_TRACK_WIDTH_M, 8.0)

# closest_point_on_polyline()'s distance_along result is used every poll to
# drive cross-track error and (for open tracks) the shortfall/finish check,
# so it needs to track the car's actual progress along the lap rather than
# re-finding "the globally nearest point on the whole track" from scratch
# each time. A figure-8 (or any track whose path crosses itself) can have
# two arc-length values that sit at, or very near, the exact same physical
# (x, y) - a fresh global search has no way to tell those apart and can
# silently snap onto the wrong pass, especially while the car is mid
# cross-track excursion near the crossing point. Once a real distance_along
# is known, later polls only need to search near it. This window has to be
# generous enough to cover a normal cross-track excursion plus however far
# the car can travel in one poll interval, but small compared to the
# arc-length gap between any two self-intersections on a real track (e.g.
# ~52 m apart on the figure-8 sample), or it would defeat the point.
TRACK_PROGRESS_SEARCH_WINDOW_M = 20.0

# --- Bayesian Optimization (BO) Settings ---
BO_NUM_INITIAL_SAMPLES = 8     
BO_TOTAL_ITERATIONS = 100      
BO_MATERN_NU = 1.5             
BO_RESTARTS = 5                
BO_RANDOM_STATE = 42           
BO_LCB_KAPPA = 1.96            

BO_EARLY_STOPPING_PATIENCE = 15       
BO_EARLY_STOPPING_MIN_DELTA = 0.01    

BO_INITIAL_GUESS = [
    int((FRONT_SPRING_MIN + FRONT_SPRING_MAX) / 2),
    int((FRONT_DAMPING_MIN + FRONT_DAMPING_MAX) / 2),
    int((REAR_SPRING_MIN + REAR_SPRING_MAX) / 2),
    int((REAR_DAMPING_MIN + REAR_DAMPING_MAX) / 2)
]


# ==========================================
# 1. THE PHYSICS SIMULATOR (Half-Car Pitch Model)
# ==========================================
class HalfCarPitchSimulator:
    """The core physics model: a 2D "half-car" (bounce + pitch, no roll)
    driving over a single road bump. simulate_bump() below is the one
    method everything else in this file calls; the rest of the class is
    the math it depends on, in the order it's used."""

    def __init__(self):
        self.chassis_sprung_mass = CHASSIS_SPRUNG_MASS
        self.front_unsprung_mass = FRONT_UNSPRUNG_MASS
        self.rear_unsprung_mass = REAR_UNSPRUNG_MASS
        self.pitch_inertia = PITCH_INERTIA
        self.distance_front_axle = DISTANCE_FRONT_AXLE
        self.distance_rear_axle = DISTANCE_REAR_AXLE

    def simulate_bump(self, front_spring_rate, front_damping, rear_spring_rate, rear_damping, profile="Comfort (Smooth Ride)"):
        """Main entry point: simulate the car driving over the road bump with
        the given spring/damper settings and return (cost_score, time points,
        chassis heave response, front wheel height, rear wheel height).
        cost_score is the scalar the Bayesian Optimization loop minimizes -
        lower is a smoother/more controlled ride under the given `profile`.
        Internally this finds the pre-bump static equilibrium, integrates the
        equations of motion through the bump event, then scores the result
        against `profile`'s weighted objective terms and realism penalties."""
        # `profile` accepts either a preset name (str, looked up in
        # PROFILE_WEIGHTS) or a fully-formed weights dict (used for the
        # user-defined "Custom" objective built in the sidebar).
        def road_speed_bump(time_t):
            total_duration = (BUMP_END_TIME - BUMP_START_TIME)
            total_train_duration = total_duration * 10
    
            if BUMP_START_TIME <= time_t <= (BUMP_START_TIME + total_train_duration):
                elapsed = time_t - BUMP_START_TIME
                bump_index = int(elapsed // total_duration)
                if bump_index >= 10:
                    return 0.0
            
                # Use modulo to get the repeating local time inside the current bump
                local_t = elapsed % total_duration
                return BUMP_HEIGHT * np.sin(np.pi * local_t / total_duration)
            return 0.0

        def road_speed_bump_velocity(time_t):
            total_duration = (BUMP_END_TIME - BUMP_START_TIME)
            total_train_duration = total_duration * 10
    
            if BUMP_START_TIME <= time_t <= (BUMP_START_TIME + total_train_duration):
                elapsed = time_t - BUMP_START_TIME
                bump_index = int(elapsed // total_duration)
                if bump_index >= 10:
                    return 0.0
            
                local_t = elapsed % total_duration
                omega = np.pi / total_duration
                return BUMP_HEIGHT * omega * np.cos(omega * local_t)
            return 0.0

        static_vertical_bounce, static_pitch_angle, static_front_wheel, static_rear_wheel = self.compute_static_equilibrium(
            front_spring_rate, rear_spring_rate
        )
        
        initial_state = [
            static_vertical_bounce, 0, 
            static_pitch_angle, 0,
            static_front_wheel, 0, 
            static_rear_wheel, 0
        ]
        time_span = (SIM_TIME_START, SIM_TIME_END)
        time_evaluation_points = np.linspace(SIM_TIME_START, SIM_TIME_END, SIM_TIME_POINTS)
        
        simulation_solution = solve_ivp(
                lambda time_t, state_vector: self.calculate_system_equations(
                time_t, state_vector, front_spring_rate, front_damping, rear_spring_rate, rear_damping, road_speed_bump, road_speed_bump_velocity
            ),
            time_span, initial_state, t_eval=time_evaluation_points,
            max_step=SINGLE_BUMP_DURATION / 4.0
        )
        
        chassis_heave_response = simulation_solution.y[0] - static_vertical_bounce
        front_height_response = simulation_solution.y[4] - static_front_wheel
        rear_height_response = simulation_solution.y[6] - static_rear_wheel
        chassis_heave_velocity = simulation_solution.y[1]
        chassis_pitch_velocity = simulation_solution.y[3]
        front_wheel_displacement = simulation_solution.y[4]
        front_wheel_velocity = simulation_solution.y[5]
        rear_wheel_displacement = simulation_solution.y[6]
        rear_wheel_velocity = simulation_solution.y[7]

        pitch_accelerations = np.gradient(chassis_pitch_velocity, time_evaluation_points)
        heave_accelerations = np.gradient(chassis_heave_velocity, time_evaluation_points)

        bump_event_mask = (time_evaluation_points >= BUMP_START_TIME) & (time_evaluation_points <= BUMP_EVENT_END)
        post_bump_mask = time_evaluation_points >= POST_BUMP_START
        post_bump_heave = chassis_heave_response[post_bump_mask]

        peak_pitch_accel = np.max(np.abs(pitch_accelerations[bump_event_mask]))
        # RMS (not just peak) pitch accel across the bump event - see the
        # "Ride Comfort Note" above for why a peak alone under-captures harshness.
        bump_pitch_accel_rms = np.sqrt(np.mean(pitch_accelerations[bump_event_mask] ** 2))

        post_heave_rms = np.sqrt(np.mean(post_bump_heave**2))
        post_pitch_vel_rms = np.sqrt(np.mean(chassis_pitch_velocity[post_bump_mask]**2))
        post_heave_accel_rms = np.sqrt(np.mean(heave_accelerations[post_bump_mask]**2))
        post_heave_peak_to_peak = (
            np.max(post_bump_heave) - np.min(post_bump_heave) if post_bump_heave.size else 0.0
        )

        # Oscillation persistence: split the post-bump window in half and
        # compare late-window heave RMS to early-window heave RMS. A setup
        # that's still shaking just as hard in the second half as the first
        # (ratio near 1, or above 1 if it's actually growing) is "a ride that
        # never settles down" - exactly the failure mode a single RMS or
        # peak-to-peak number averaged over the *whole* post-bump window can
        # miss, because a quick sharp decay and a low-amplitude-but-endless
        # ring can land on similar RMS values despite being very different
        # rides. This is dimensionless by construction (no REF_* needed).
        post_bump_times = time_evaluation_points[post_bump_mask]
        if post_bump_times.size >= 2:
            post_bump_midpoint = (post_bump_times[0] + post_bump_times[-1]) / 2.0
            early_half_heave = post_bump_heave[post_bump_times <= post_bump_midpoint]
            late_half_heave = post_bump_heave[post_bump_times > post_bump_midpoint]
            early_half_rms = np.sqrt(np.mean(early_half_heave ** 2)) if early_half_heave.size else 0.0
            late_half_rms = np.sqrt(np.mean(late_half_heave ** 2)) if late_half_heave.size else 0.0
            post_bump_persistence_ratio = late_half_rms / (early_half_rms + 1e-6)
        else:
            post_bump_persistence_ratio = 0.0

        # Dynamic tire load variation during the bump event: how far tire
        # force swings away from its static (pre-bump) value. Lower swing
        # means the contact patch stays more consistently loaded, which is
        # what actually matters for grip - this is what the Track profile's
        # stiffness/damping choices should be optimizing for, rather than
        # rewarding raw stiffness on its own.
        road_height_front_series = np.array([road_speed_bump(t) for t in time_evaluation_points])
        road_height_rear_series = np.array([road_speed_bump(t - REAR_AXLE_ROAD_LAG) for t in time_evaluation_points])
        road_vel_front_series = np.array([road_speed_bump_velocity(t) for t in time_evaluation_points])
        road_vel_rear_series = np.array([road_speed_bump_velocity(t - REAR_AXLE_ROAD_LAG) for t in time_evaluation_points])

        front_tire_force_series = np.maximum(
            0.0,
            TIRE_STIFFNESS * (road_height_front_series - front_wheel_displacement)
            + TIRE_DAMPING * (road_vel_front_series - front_wheel_velocity),
        )
        rear_tire_force_series = np.maximum(
            0.0,
            TIRE_STIFFNESS * (road_height_rear_series - rear_wheel_displacement)
            + TIRE_DAMPING * (road_vel_rear_series - rear_wheel_velocity),
        )
        static_front_tire_force = max(0.0, TIRE_STIFFNESS * (0.0 - static_front_wheel))
        static_rear_tire_force = max(0.0, TIRE_STIFFNESS * (0.0 - static_rear_wheel))

        front_tire_load_deviation = front_tire_force_series[bump_event_mask] - static_front_tire_force
        rear_tire_load_deviation = rear_tire_force_series[bump_event_mask] - static_rear_tire_force
        tire_load_variation_rms = np.sqrt(
            np.mean(front_tire_load_deviation ** 2 + rear_tire_load_deviation ** 2)
        )

        weights = profile if isinstance(profile, dict) else PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["Balanced"])
        cost_score = (
            weights["peak_pitch_accel"] * (peak_pitch_accel / REF_PEAK_PITCH_ACCEL)
            + weights.get("bump_pitch_accel_rms", 0.0) * (bump_pitch_accel_rms / REF_BUMP_PITCH_ACCEL_RMS)
            + weights["post_heave_rms"] * (post_heave_rms / REF_POST_HEAVE_RMS)
            + weights["post_pitch_vel_rms"] * (post_pitch_vel_rms / REF_POST_PITCH_VEL_RMS)
            + weights["post_heave_accel_rms"] * (post_heave_accel_rms / REF_POST_HEAVE_ACCEL_RMS)
            + weights["post_heave_peak_to_peak"] * (post_heave_peak_to_peak / REF_POST_HEAVE_PEAK_TO_PEAK)
            + weights.get("oscillation_persistence", 0.0) * post_bump_persistence_ratio
            + weights.get("tire_load_variation", 0.0) * (tire_load_variation_rms / REF_TIRE_LOAD_VARIATION)
            + weights["spring_rate_sum"] * ((front_spring_rate + rear_spring_rate) / REF_SPRING_RATE_SUM)
            + self.compute_realism_penalties(
                front_spring_rate, front_damping, rear_spring_rate, rear_damping, weights
            )
        )
        
        return cost_score, time_evaluation_points, chassis_heave_response, front_height_response, rear_height_response

    def compute_static_equilibrium(self, front_spring_rate, rear_spring_rate):
        """Find the car's resting position (ride height + pitch) under its own
        weight, before the bump event, by solving for the state where every
        net force/torque is zero. simulate_bump() starts its time integration
        from this equilibrium rather than from an arbitrary zero state."""
        def equilibrium_equations(state):
            vertical_bounce, pitch_angle, front_wheel_displacement, rear_wheel_displacement = state
            
            chassis_front_displacement = vertical_bounce - self.distance_front_axle * pitch_angle
            chassis_rear_displacement = vertical_bounce + self.distance_rear_axle * pitch_angle
            
            front_compression = front_wheel_displacement - chassis_front_displacement
            rear_compression = rear_wheel_displacement - chassis_rear_displacement
            
            front_suspension_force = front_spring_rate * front_compression
            rear_suspension_force = rear_spring_rate * rear_compression
            
            front_tire_force = max(0.0, TIRE_STIFFNESS * (0.0 - front_wheel_displacement))
            rear_tire_force = max(0.0, TIRE_STIFFNESS * (0.0 - rear_wheel_displacement))
            
            pitch_torque = (rear_suspension_force * self.distance_rear_axle - front_suspension_force * self.distance_front_axle)
            
            return [
                (front_suspension_force + rear_suspension_force) / self.chassis_sprung_mass - GRAVITY,
                pitch_torque, 
                (front_tire_force - front_suspension_force) / self.front_unsprung_mass - GRAVITY,
                (rear_tire_force - rear_suspension_force) / self.rear_unsprung_mass - GRAVITY,
            ]

        solution = root(equilibrium_equations, [0.0, 0.0, -0.01, -0.01])
        return solution.x 

    def calculate_system_equations(self, time_t, state_vector, front_spring_rate, front_damping, rear_spring_rate, rear_damping, road_profile_func, road_velocity_func):
        """The ODE right-hand side (F = m*a for each of the 4 masses: chassis
        bounce, chassis pitch, front wheel, rear wheel) that simulate_bump()
        hands to scipy's solve_ivp to integrate the car's motion through the
        bump event. Returns the state vector's time-derivative at time_t."""
        (vertical_bounce, vertical_velocity, 
         pitch_angle, pitch_velocity, 
         front_wheel_displacement, front_wheel_velocity, 
         rear_wheel_displacement, rear_wheel_velocity) = state_vector
        
        road_height_front = road_profile_func(time_t)
        road_height_rear = road_profile_func(time_t - REAR_AXLE_ROAD_LAG)
        road_vel_front = road_velocity_func(time_t)
        road_vel_rear = road_velocity_func(time_t - REAR_AXLE_ROAD_LAG)
        
        chassis_front_displacement = vertical_bounce - self.distance_front_axle * pitch_angle
        chassis_rear_displacement = vertical_bounce + self.distance_rear_axle * pitch_angle
        chassis_front_velocity = vertical_velocity - self.distance_front_axle * pitch_velocity
        chassis_rear_velocity = vertical_velocity + self.distance_rear_axle * pitch_velocity
        
        front_compression = front_wheel_displacement - chassis_front_displacement
        rear_compression = rear_wheel_displacement - chassis_rear_displacement
        
        front_comp_velocity = front_wheel_velocity - chassis_front_velocity
        rear_comp_velocity = rear_wheel_velocity - chassis_rear_velocity
        
        front_suspension_force = front_spring_rate * front_compression + front_damping * front_comp_velocity
        rear_suspension_force = rear_spring_rate * rear_compression + rear_damping * rear_comp_velocity
        
        front_tire_compression = road_height_front - front_wheel_displacement
        rear_tire_compression = road_height_rear - rear_wheel_displacement
        
        front_tire_comp_velocity = road_vel_front - front_wheel_velocity
        rear_tire_comp_velocity = road_vel_rear - rear_wheel_velocity
        
        front_tire_force = max(0.0, TIRE_STIFFNESS * front_tire_compression + TIRE_DAMPING * front_tire_comp_velocity)
        rear_tire_force = max(0.0, TIRE_STIFFNESS * rear_tire_compression + TIRE_DAMPING * rear_tire_comp_velocity)
        
        vertical_chassis_acceleration = (front_suspension_force + rear_suspension_force) / self.chassis_sprung_mass - GRAVITY
        pitch_chassis_acceleration = (rear_suspension_force * self.distance_rear_axle - front_suspension_force * self.distance_front_axle) / self.pitch_inertia 
        
        front_wheel_acceleration = (front_tire_force - front_suspension_force) / self.front_unsprung_mass - GRAVITY
        rear_wheel_acceleration = (rear_tire_force - rear_suspension_force) / self.rear_unsprung_mass - GRAVITY
        
        return [
            vertical_velocity, vertical_chassis_acceleration, 
            pitch_velocity, pitch_chassis_acceleration, 
            front_wheel_velocity, front_wheel_acceleration, 
            rear_wheel_velocity, rear_wheel_acceleration
        ]

    def compute_realism_penalties(self, front_spring_rate, front_damping, rear_spring_rate, rear_damping, weights):
        """Penalty terms that keep the optimizer from picking physically
        unrealistic setups that would otherwise score well on ride-response
        alone: springs too soft (underspring_penalty) or a damping ratio far
        from the profile's target band (damping_ratio_penalty). Added into
        simulate_bump()'s cost_score, weighted by `weights`."""
        # Each profile carries its own target/floor now (see PROFILE_WEIGHTS),
        # rather than every profile sharing one hardcoded comfort-tuned value.
        spring_rate_min = weights.get("spring_rate_min", 25000.0)
        damping_ratio_target = weights.get("damping_ratio_target", 0.45)
        damping_ratio_tolerance = weights.get("damping_ratio_tolerance", 0.20)

        # Underspring penalty: normalized as a fractional shortfall relative
        # to the floor, so it's O(1) at "zero stiffness" and O(0) at or above
        # the floor, regardless of what the floor's absolute value is.
        front_shortfall_frac = max(0.0, spring_rate_min - front_spring_rate) / spring_rate_min
        rear_shortfall_frac = max(0.0, spring_rate_min - rear_spring_rate) / spring_rate_min
        underspring_penalty = front_shortfall_frac ** 2 + rear_shortfall_frac ** 2

        # Damping-ratio penalty: normalized in units of "tolerance widths"
        # past the allowed band around the target, so weight values mean
        # the same thing across profiles even though targets/tolerances differ.
        front_damping_ratio, rear_damping_ratio = self.compute_damping_ratios(
            front_spring_rate, front_damping, rear_spring_rate, rear_damping
        )
        damping_ratio_penalty = 0.0
        for damping_ratio in (front_damping_ratio, rear_damping_ratio):
            excess_deviation = abs(damping_ratio - damping_ratio_target) - damping_ratio_tolerance
            if excess_deviation > 0.0:
                # Cap "tolerance widths past the band" before squaring - see
                # DAMPING_RATIO_PENALTY_MAX_WIDTHS above. Without this, an
                # overdamped setup (damping large relative to a soft spring's
                # small critical damping) has no upper bound on how far past
                # the band it can sit, and this term alone could reach the
                # hundreds - dwarfing every actual ride-response measurement
                # in the cost and making the score reflect "how close to my
                # assumed target damping ratio" rather than "how the car
                # actually rides".
                widths_past_band = min(
                    excess_deviation / damping_ratio_tolerance, DAMPING_RATIO_PENALTY_MAX_WIDTHS
                )
                damping_ratio_penalty += widths_past_band ** 2

        return (
            weights.get("underspring_penalty", 0.0) * underspring_penalty
            + weights.get("damping_ratio_penalty", 0.0) * damping_ratio_penalty
        )

    def compute_damping_ratios(self, front_spring_rate, front_damping, rear_spring_rate, rear_damping):
        """Convert raw spring/damping rates into dimensionless damping ratios
        (actual damping / critical damping) for front and rear, used by
        compute_realism_penalties() to check against each profile's target
        damping-ratio band."""
        front_sprung_load = self.chassis_sprung_mass * self.distance_rear_axle / WHEELBASE
        rear_sprung_load = self.chassis_sprung_mass * self.distance_front_axle / WHEELBASE
        front_critical_damping = 2.0 * np.sqrt(front_spring_rate * front_sprung_load)
        rear_critical_damping = 2.0 * np.sqrt(rear_spring_rate * rear_sprung_load)
        front_damping_ratio = front_damping / front_critical_damping
        rear_damping_ratio = rear_damping / rear_critical_damping
        return front_damping_ratio, rear_damping_ratio


# --- Scoring & plotting helpers used by the Streamlit UI (section 3) to
# explain/display simulate_bump()'s results - not called by the simulator
# or optimizer themselves. ---
def resolve_profile_weights(profile):
    """Same lookup simulate_bump() uses internally, exposed for display purposes
    (e.g. the score-explanation popover) so the two never drift apart."""
    return profile if isinstance(profile, dict) else PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["Balanced"])


def render_score_explanation(weights, profile_name):
    """Renders a breakdown of exactly which terms feed the cost score for the
    given profile/weights, reusing the same (key, label, ..., help) metadata
    that drives the Custom-profile sidebar inputs so this can't fall out of
    sync with what CUSTOM_PROFILE_OBJECTIVE_FIELDS/REALISM_FIELDS describe."""
    st.markdown(f"**How the \u201c{profile_name}\u201d score is calculated**")
    st.caption(
        "Every raw simulated quantity (a peak, an RMS, a peak-to-peak swing, etc.) "
        "is first divided by a reference value to turn it into a dimensionless, "
        "roughly 0-5-scale number, then multiplied by the weight shown below. "
        "The final score is just the sum of all of these, plus the realism "
        "penalties at the bottom. **A weight of 0 means that term has no effect "
        "on this profile's score at all.**"
    )

    st.markdown("**Ride objective terms**")
    for field_key, field_label, *_bounds, field_help in CUSTOM_PROFILE_OBJECTIVE_FIELDS:
        weight_value = weights.get(field_key, 0.0)
        marker = "🔹" if weight_value > 0 else "⚪"
        st.markdown(f"{marker} **{field_label}** — weight `{weight_value:g}`")
        st.caption(field_help)

    st.markdown("**Realism penalties**")
    st.caption(
        "These discourage physically-degenerate setups (near-zero springs or a "
        "wildly wrong damping ratio) rather than measuring ride comfort directly. "
        "The damping-ratio penalty is capped at "
        f"{DAMPING_RATIO_PENALTY_MAX_WIDTHS:g} tolerance-widths past the target "
        "band before it's squared, so one extreme setup can't blow up and swamp "
        "every other term."
    )
    for field_key, field_label, *_bounds, field_help in CUSTOM_PROFILE_REALISM_FIELDS:
        value = weights.get(field_key, 0.0)
        if field_key.endswith("_penalty"):
            marker = "🔹" if value > 0 else "⚪"
            st.markdown(f"{marker} **{field_label}** — weight `{value:g}`")
        else:
            st.markdown(f"◦ **{field_label}** — currently `{value:g}`")
        st.caption(field_help)


def plot_optimization_history(history):
    """Render the 3-panel live progress chart (score per trial, best-so-far,
    and parameter values over trials) shown while the Bayesian Optimization
    loop is running. `history` is the dict the optimization loop appends to
    after every simulator evaluation."""
    iterations = history["iteration"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].plot(iterations, history["score"], "o", alpha=0.45, markersize=3, color="steelblue", label="Trial score")
    axes[0].plot(iterations, history["best_score"], "-", linewidth=2, color="forestgreen", label="Best so far")
    axes[0].set_xlabel("Evaluation #")
    axes[0].set_ylabel("Cost Score")
    axes[0].set_title("Cost Score")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(iterations, history["front_spring"], label="Front Spring", color="crimson")
    axes[1].plot(iterations, history["rear_spring"], label="Rear Spring", color="darkorange", linestyle="--")
    axes[1].set_xlabel("Evaluation #")
    axes[1].set_ylabel("Spring Rate (N/m)")
    axes[1].set_title("Spring Rates")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(iterations, history["front_damping"], label="Front Damping", color="royalblue")
    axes[2].plot(iterations, history["rear_damping"], label="Rear Damping", color="teal", linestyle="--")
    axes[2].set_xlabel("Evaluation #")
    axes[2].set_ylabel("Damping (Ns/m)")
    axes[2].set_title("Damping")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


# ==========================================
# 1.5 TRACK / CONE PATH GEOMETRY
# ==========================================
# Pure functions, no Streamlit/beamngpy dependency - this is deliberate (see
# TRACK_CONE_AI_INTEGRATION.md §8 "Verification plan"), so this whole block
# can be exercised in a standalone script the same way HANDOFF.md describes
# doing for the suspension cost function, before ever touching the live game.
#
# TrackPoint/TrackDefinition come first, before any function that uses them -
# they're simple data classes referenced as type hints everywhere below, so
# (like a struct) they need to exist before the functions that mention them
# are even defined. After that, functions are ordered top-down by entry
# point first (load_track_file, generate_all_cone_positions, build_ai_script,
# compute_finish_line, select_finish_checkpoints - the ones the Streamlit UI
# and the BeamNG track bridge actually call directly) followed by the
# lower-level geometry helpers each of those relies on.

@dataclass
class TrackPoint:
    """One point along a track centerline, in the target level's world-space
    coordinate frame (same frame as BEAMNG_SPAWN_POS)."""
    x: float
    y: float
    z: Optional[float] = None
    width_m: Optional[float] = None
    speed_mps: Optional[float] = None


@dataclass
class TrackDefinition:
    name: str
    points: list
    closed_loop: bool = False
    default_width_m: float = DEFAULT_TRACK_WIDTH_M
    default_speed_mps: float = DEFAULT_TRACK_SPEED_MPS


def load_track_file(uploaded_file, filename_hint=None, origin_xy=TRACK_WORLD_ORIGIN_XY):
    """Normalize a CSV or JSON upload into a validated TrackDefinition.
    Raises ValueError with a user-facing message on any validation failure -
    callers should catch this and show it via st.error rather than letting
    it surface as a raw traceback, since a malformed track file is an
    expected user-facing error case, not a bug.

    origin_xy translates every point's (x, y) by this much after parsing,
    converting a track authored in its own local frame (see
    TRACK_WORLD_ORIGIN_XY above) into BeamNG world coordinates. z is left
    untouched - track files already specify a real ground height there.
    """
    name_for_ext = (filename_hint or getattr(uploaded_file, "name", "") or "").lower()
    is_json = name_for_ext.endswith(".json")

    if is_json:
        raw = parse_track_json(uploaded_file)
        raw_points = raw.get("points", [])
        track_name = raw.get("name", "uploaded_track")
        closed_loop = bool(raw.get("closed_loop", False))
        default_width = float(raw.get("default_width_m", DEFAULT_TRACK_WIDTH_M))
        default_speed = float(raw.get("default_speed_mps", DEFAULT_TRACK_SPEED_MPS))
    else:
        raw_points = parse_track_csv(uploaded_file)
        track_name = filename_hint or "uploaded_track"
        closed_loop = False
        default_width = DEFAULT_TRACK_WIDTH_M
        default_speed = DEFAULT_TRACK_SPEED_MPS

    if len(raw_points) < 3:
        raise ValueError(
            f"Track file has {len(raw_points)} point(s) - at least 3 are "
            "required (this is also BeamNG's own execute_script() minimum "
            "for a valid AI path, see TRACK_CONE_AI_INTEGRATION.md §4)."
        )

    points = []
    for i, p in enumerate(raw_points):
        try:
            x = float(p["x"])
            y = float(p["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Track point {i} is missing a valid x/y value: {p}") from exc
        z = float(p["z"]) if p.get("z") not in (None, "") else None
        width_m = float(p["width_m"]) if p.get("width_m") not in (None, "") else None
        speed_mps = float(p["speed_mps"]) if p.get("speed_mps") not in (None, "") else None
        points.append(TrackPoint(
            x=x + origin_xy[0], y=y + origin_xy[1],
            z=z, width_m=width_m, speed_mps=speed_mps,
        ))

    deduped = _dedupe_near_points(points, MIN_TRACK_POINT_SPACING_M)
    if len(deduped) < 3:
        raise ValueError(
            "Fewer than 3 points remain after removing near-duplicate "
            f"points (closer than {MIN_TRACK_POINT_SPACING_M} m apart) - "
            "check the source file for accidental duplicate rows."
        )

    return TrackDefinition(
        name=track_name,
        points=deduped,
        closed_loop=closed_loop,
        default_width_m=default_width,
        default_speed_mps=default_speed,
    )


def generate_all_cone_positions(track: TrackDefinition, cone_spacing_m=DEFAULT_CONE_SPACING_M,
                                 run_up_m=RUN_UP_DISTANCE_M):
    """The actual cone layout used for spawning/preview: the normal
    track-boundary cones (generate_cone_positions()) plus a matching set
    along the run-up lane (generate_run_up_cone_positions()), with any
    track-boundary cone that geometrically falls inside the run-up lane
    dropped first.

    That last part matters on tracks where some other stretch of the path
    (e.g. the return leg of a closed loop, or a tight figure-8 crossing
    near its own start) happens to pass close to the run-up lane in world
    space - generate_cone_positions() has no notion of the run-up lane at
    all, so nothing before this stopped it from placing a boundary cone
    directly in the car's path before it even reaches node 0. The check is
    purely geometric (is this cone inside the run-up lane's rectangle?),
    not "which track point produced it," so it catches that case
    regardless of which part of the track a blocking cone came from.

    A cone sitting almost exactly at the track's own first point is left
    alone even though it's technically at the far end of the lane - that's
    the legitimate start-gate cone pair, not something blocking the
    approach to it.
    """
    track_cones = generate_cone_positions(track, cone_spacing_m=cone_spacing_m)
    geom = compute_run_up_geometry(track, run_up_m=run_up_m)
    start_x, start_y, _ = geom["run_up_start"]
    ux, uy = geom["tangent"]
    half = geom["width_m"] / 2.0
    # Stay a bit short of the far end of the lane (run_up_m) so the
    # legitimate start-gate cones - which sit right at that end - aren't
    # mistaken for something blocking the approach to them.
    along_limit = max(0.0, run_up_m - min(cone_spacing_m / 2.0, run_up_m))

    def _blocks_run_up(cone):
        dx, dy = cone["x"] - start_x, cone["y"] - start_y
        along = dx * ux + dy * uy
        lateral = -dx * uy + dy * ux
        return -1e-6 <= along <= along_limit and abs(lateral) <= half

    filtered_track_cones = [c for c in track_cones if not _blocks_run_up(c)]
    run_up_cones = generate_run_up_cone_positions(track, run_up_m=run_up_m, cone_spacing_m=cone_spacing_m)
    return filtered_track_cones + run_up_cones


def build_ai_script(track: TrackDefinition, resample_spacing_m=TRACK_AI_SCRIPT_RESAMPLE_SPACING_M,
                     min_dt=MIN_SCRIPT_DT, run_up_m=0.0):
    """Build the timestamped waypoint list for vehicle.ai.execute_script()
    (see TRACK_CONE_AI_INTEGRATION.md §4). Resamples the centerline at
    resample_spacing_m first - deliberately coarser than, and independent
    of, cone spacing (a dense cone-spacing-resolution polyline is the wrong
    input here). Requires at least 3 output nodes, matching execute_script's
    own hard minimum.

    run_up_m, if > 0, prepends one extra node run_up_m behind the first
    point along the initial tangent (see compute_run_up_start()) so the
    script's own timeline agrees with where the car is actually spawned -
    without this the vehicle would spawn behind node 0 but the script
    would still expect it to already be AT node 0 at t=0.
    """
    resampled = resample_track_by_arc_length(track, resample_spacing_m)
    if track.closed_loop and resampled:
        resampled = resampled + [resampled[0]]
    if len(resampled) < 3:
        raise ValueError(
            f"Only {len(resampled)} node(s) after resampling at "
            f"{resample_spacing_m} m spacing - need at least 3 for "
            "execute_script(). Try a shorter track or a smaller resample spacing."
        )
    script = []
    t = 0.0
    if run_up_m > 0:
        run_up_pos, _ = compute_run_up_start(track, run_up_m=run_up_m)
        speed = resampled[0].speed_mps or track.default_speed_mps or DEFAULT_TRACK_SPEED_MPS
        dt = max(run_up_m / max(speed, 0.1), min_dt)
        script.append({"x": run_up_pos[0], "y": run_up_pos[1], "z": run_up_pos[2] or 0.0, "t": t})
        t += dt
    script.append({"x": resampled[0].x, "y": resampled[0].y, "z": resampled[0].z or 0.0, "t": t})
    for prev, cur in zip(resampled, resampled[1:]):
        dist = math.hypot(cur.x - prev.x, cur.y - prev.y)
        speed = cur.speed_mps or track.default_speed_mps or DEFAULT_TRACK_SPEED_MPS
        dt = max(dist / max(speed, 0.1), min_dt)
        t += dt
        script.append({"x": cur.x, "y": cur.y, "z": cur.z or 0.0, "t": t})
    return script


def compute_finish_line(track: TrackDefinition):
    """Defines the start/finish line(s) used to auto-stop a track run (see
    run_beamng_track_test() below) - this is what was missing before: the
    old stopping condition was just a centerline distance-along-track
    threshold, which only worked for open tracks and reads confusingly once
    the car has overcorrected/wandered (exactly what was being seen).

    The start line sits at the track's first point, perpendicular to the
    initial tangent - the same tangent compute_run_up_start() spawns the
    car along, so the car is guaranteed to cross it once early on just by
    driving off the run-up.

    Closed loops reuse that same line as the finish line: one full lap is
    "cross it once to actually start, cross it a second time to finish."
    Open tracks get a separate finish line at the *last* point, using the
    tangent of the final segment, so completion doesn't depend on the
    noisier distance-along-track heuristic at all.

    Returns a dict: start_point/start_tangent, finish_point/finish_tangent
    ((x, y) tuples / (ux, uy) unit tangents), and same_line (True for
    closed loops, meaning the finish line above is literally the start
    line reused).
    """
    p0, p1 = track.points[0], track.points[1]
    start_point = (p0.x, p0.y)
    start_tangent = _unit_tangent(p0, p1)
    if track.closed_loop:
        finish_point, finish_tangent, same_line = start_point, start_tangent, True
    else:
        p_last, p_prev = track.points[-1], track.points[-2]
        finish_point = (p_last.x, p_last.y)
        finish_tangent = _unit_tangent(p_prev, p_last)
        same_line = False
    return {
        "start_point": start_point, "start_tangent": start_tangent,
        "finish_point": finish_point, "finish_tangent": finish_tangent,
        "same_line": same_line,
    }


def select_finish_checkpoints(track: TrackDefinition, cone_positions, count=FINISH_CHECKPOINT_COUNT,
                               rng=None):
    """Pick `count` checkpoints spread evenly around the track's arc length,
    each one an actual cone drawn from cone_positions (i.e. "randomly
    selected evenly spaced cones from the file") rather than a synthetic
    point - see FINISH_CHECKPOINT_COUNT above for why this exists: a
    closed-loop track's finish gate is the same physical line as its start
    gate, and the car can cross it more than once near the very start of a
    run (run-up alignment wobble) without having driven any of the actual
    lap. Requiring these to all be passed first before a gate crossing
    counts as the finish rules that out, since they're spread around the
    rest of the track and can't be reached that way.

    Splits [0, total_track_length) into `count` equal arc-length bins and
    picks one cone at random from whichever cones happen to fall in each
    bin (falling back to the single closest cone to that bin's center if
    none do) - "randomly selected" so the checkpoint isn't always the same
    cone (or always on the same left/right side) on repeated runs of the
    same track, while "evenly spaced" keeps them spread around the whole
    lap instead of clustering wherever cone density happens to be higher.

    Returns a list of {"x", "y", "along_m"} dicts. Empty if there are no
    cones to pick from or the track has zero length."""
    if not cone_positions or count <= 0:
        return []
    total_length = _cumulative_arc_length(track.points, track.closed_loop)[1][-1]
    if total_length <= 0:
        return []
    rng = rng if rng is not None else random

    # Tag each cone with its arc-length position along the centerline so
    # cones (which live on the left/right boundary offsets, not the
    # centerline itself - see generate_cone_positions()) can be grouped
    # into arc-length bins.
    tagged = [
        (closest_point_on_polyline(cone["x"], cone["y"], track)[0], cone)
        for cone in cone_positions
    ]

    bin_width = total_length / count
    checkpoints = []
    for i in range(count):
        bin_lo, bin_hi = i * bin_width, (i + 1) * bin_width
        candidates = [cone for along, cone in tagged if bin_lo <= along < bin_hi]
        if not candidates:
            # No cone happened to land in this bin (possible with sparse
            # cone spacing relative to `count`) - fall back to whichever
            # cone is arc-length-closest to the bin's center so every bin
            # still contributes one checkpoint.
            bin_center = (bin_lo + bin_hi) / 2.0
            _, nearest_cone = min(tagged, key=lambda t: abs(t[0] - bin_center))
            candidates = [nearest_cone]
        chosen = rng.choice(candidates)
        along, _, _ = closest_point_on_polyline(chosen["x"], chosen["y"], track)
        checkpoints.append({"x": chosen["x"], "y": chosen["y"], "along_m": along})
    return checkpoints


def _dedupe_near_points(points, min_spacing_m):
    """Drop consecutive points closer together than min_spacing_m. See
    TRACK_CONE_AI_INTEGRATION.md §1 - near-duplicate points produce a
    near-zero-duration script segment that execute_script cannot satisfy
    physically (it starts skipping points to catch up)."""
    if not points:
        return points
    cleaned = [points[0]]
    for p in points[1:]:
        prev = cleaned[-1]
        dist = math.hypot(p.x - prev.x, p.y - prev.y)
        if dist >= min_spacing_m:
            cleaned.append(p)
    return cleaned


def parse_track_csv(file_like):
    """Parse a CSV track file (see TRACK_CONE_AI_INTEGRATION.md §1 for the
    schema). Returns a plain list of raw dict rows - load_track_file() below
    is what turns this into a validated TrackDefinition."""
    text = file_like.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        rows.append({k: v for k, v in row.items() if v not in (None, "")})
    return rows


def parse_track_json(file_like):
    """Parse a JSON track file (see TRACK_CONE_AI_INTEGRATION.md §1 for the
    schema). Returns the parsed dict as-is - load_track_file() normalizes it."""
    text = file_like.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return json.loads(text)


def generate_cone_positions(track: TrackDefinition, cone_spacing_m=DEFAULT_CONE_SPACING_M):
    """Full pipeline from TRACK_CONE_AI_INTEGRATION.md §2: derive left/right
    boundaries from the centerline, then arc-length resample each boundary
    independently at cone_spacing_m. Returns a flat list of dicts:
    {"x", "y", "z", "side"} with side in {"left", "right"}."""
    left_pts, right_pts = compute_boundary_offsets(track)
    left_track = TrackDefinition(
        name=track.name + "_left", points=left_pts, closed_loop=track.closed_loop,
        default_width_m=0.0, default_speed_mps=track.default_speed_mps,
    )
    right_track = TrackDefinition(
        name=track.name + "_right", points=right_pts, closed_loop=track.closed_loop,
        default_width_m=0.0, default_speed_mps=track.default_speed_mps,
    )
    cones = []
    for side, side_track in (("left", left_track), ("right", right_track)):
        for p in resample_track_by_arc_length(side_track, cone_spacing_m):
            cones.append({"x": p.x, "y": p.y, "z": p.z or 0.0, "side": side})
    return cones


def resample_track_by_arc_length(track: TrackDefinition, spacing_m):
    """Arc-length resample of the centerline at a fixed spacing, independent
    of input point density (see TRACK_CONE_AI_INTEGRATION.md §2 step 4 /
    §4). Always keeps the very first point exactly (and the closing point
    for a closed loop) so the resampled path starts where the input did."""
    pts, cumulative = _cumulative_arc_length(track.points, track.closed_loop)
    total = cumulative[-1]
    if total <= 0:
        return list(track.points)
    n_steps = max(1, int(total // spacing_m))
    resampled = [
        _interpolate_track_point(pts, cumulative, i * (total / n_steps), track.default_speed_mps)
        for i in range(n_steps)
    ]
    if not track.closed_loop:
        resampled.append(
            _interpolate_track_point(pts, cumulative, total, track.default_speed_mps)
        )
    return resampled


def _interpolate_track_point(pts, cumulative, target_dist, default_speed):
    """Linearly interpolate x/y/z/speed_mps at a given arc-length distance
    along the pts/cumulative pair produced by _cumulative_arc_length()."""
    total = cumulative[-1]
    target_dist = max(0.0, min(target_dist, total))
    # Find the segment containing target_dist.
    lo = 0
    hi = len(cumulative) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if cumulative[mid] <= target_dist:
            lo = mid
        else:
            hi = mid
    seg_len = cumulative[hi] - cumulative[lo]
    t = 0.0 if seg_len < 1e-9 else (target_dist - cumulative[lo]) / seg_len
    a, b = pts[lo], pts[hi]
    x = a.x + t * (b.x - a.x)
    y = a.y + t * (b.y - a.y)
    az = a.z if a.z is not None else 0.0
    bz = b.z if b.z is not None else az
    z = az + t * (bz - az)
    aspeed = a.speed_mps if a.speed_mps is not None else default_speed
    bspeed = b.speed_mps if b.speed_mps is not None else aspeed
    speed = aspeed + t * (bspeed - aspeed)
    return TrackPoint(x=x, y=y, z=z, speed_mps=speed)


def compute_boundary_offsets(track: TrackDefinition):
    """For each centerline point, compute left/right boundary points offset
    by width_m / 2 along the local normal (see TRACK_CONE_AI_INTEGRATION.md
    §2 steps 1-3). Central difference for interior points, forward/backward
    difference at the open ends unless closed_loop."""
    pts = track.points
    n = len(pts)
    left_pts, right_pts = [], []
    for i in range(n):
        if track.closed_loop:
            prev_p, next_p = pts[(i - 1) % n], pts[(i + 1) % n]
        else:
            prev_p = pts[i - 1] if i > 0 else pts[i]
            next_p = pts[i + 1] if i < n - 1 else pts[i]
        tx, ty = next_p.x - prev_p.x, next_p.y - prev_p.y
        tlen = math.hypot(tx, ty)
        if tlen < 1e-9:
            nx, ny = 0.0, 1.0
        else:
            nx, ny = -ty / tlen, tx / tlen  # left-hand normal
        width = pts[i].width_m if pts[i].width_m is not None else track.default_width_m
        half = width / 2.0
        z = pts[i].z if pts[i].z is not None else 0.0
        left_pts.append(TrackPoint(x=pts[i].x + nx * half, y=pts[i].y + ny * half, z=z))
        right_pts.append(TrackPoint(x=pts[i].x - nx * half, y=pts[i].y - ny * half, z=z))
    return left_pts, right_pts


def generate_run_up_cone_positions(track: TrackDefinition, run_up_m=RUN_UP_DISTANCE_M,
                                    cone_spacing_m=DEFAULT_CONE_SPACING_M):
    """Cones flanking the run-up lane itself. generate_cone_positions() only
    ever walks the track's own points, so the run-up stretch the car
    actually spawns on and drives through before reaching node 0 had no
    cones of its own - this fills that gap, using the same left/right
    normal-offset convention as compute_boundary_offsets() (so a run-up
    cone and the first track-boundary cone line up on the same left/right
    lines rather than kinking sideways at the handoff point) and the same
    width as the track's own first point (so the lane doesn't visually
    pinch or flare right where the run-up meets the track).

    Stops short of the track's first point itself (that pair of cones is
    already produced by generate_cone_positions()) so the two cone sets
    don't overlap/duplicate right at the handoff.
    """
    geom = compute_run_up_geometry(track, run_up_m=run_up_m)
    ux, uy = geom["tangent"]
    nx, ny = -uy, ux  # left-hand normal, matching compute_boundary_offsets()
    half = geom["width_m"] / 2.0
    run_up_start = geom["run_up_start"]
    n_steps = max(1, int(run_up_m // cone_spacing_m))
    step = run_up_m / n_steps
    cones = []
    for i in range(n_steps):
        t = i * step
        cx = run_up_start[0] + ux * t
        cy = run_up_start[1] + uy * t
        cz = run_up_start[2] or 0.0
        cones.append({"x": cx + nx * half, "y": cy + ny * half, "z": cz, "side": "left"})
        cones.append({"x": cx - nx * half, "y": cy - ny * half, "z": cz, "side": "right"})
    return cones


def compute_run_up_geometry(track: TrackDefinition, run_up_m=RUN_UP_DISTANCE_M):
    """Shared geometry for the run-up lane - the straight stretch the car
    drives from its spawn point (see compute_run_up_start()) up to the
    track's own first point. Used by both generate_run_up_cone_positions()
    (to lay cones along this lane) and generate_all_cone_positions() (to
    filter out any main-track cone that would fall inside it). Kept as a
    separate function so both stay in agreement about where the lane
    actually is instead of recomputing the tangent/width independently."""
    p0, p1 = track.points[0], track.points[1]
    dx, dy = p1.x - p0.x, p1.y - p0.y
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    z0 = p0.z if p0.z is not None else 0.0
    run_up_start = (p0.x - ux * run_up_m, p0.y - uy * run_up_m, z0)
    width = p0.width_m if p0.width_m is not None else track.default_width_m
    return {
        "run_up_start": run_up_start,
        "track_start": (p0.x, p0.y, z0),
        "tangent": (ux, uy),
        "width_m": width,
    }


def compute_run_up_start(track: TrackDefinition, run_up_m=RUN_UP_DISTANCE_M):
    """Returns (run_up_pos, rot_quat) - a point run_up_m behind the track's
    first point along its initial tangent, and the rot_quat facing along
    that same tangent. Used so the car spawns with room to roll up to
    speed and align onto the path, instead of starting from a dead stop
    exactly on top of the first AI-script node facing an unrelated
    direction."""
    p0, p1 = track.points[0], track.points[1]
    dx, dy = p1.x - p0.x, p1.y - p0.y
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    run_up_pos = (p0.x - ux * run_up_m, p0.y - uy * run_up_m, p0.z)
    return run_up_pos, heading_to_spawn_quat(ux, uy)


def heading_to_spawn_quat(dx, dy):
    """Convert a direction vector into a spawn rot_quat using the same
    forward/backward convention BEAMNG_SPAWN_ROT_QUAT above needed an
    empirical correction for. See TRACK_HEADING_ROT_OFFSET_DEG - currently
    -90.0 to fix a reported 90-deg-counterclockwise spawn-heading error;
    adjust that constant (not this function) if the error direction ever
    changes."""
    heading_deg = math.degrees(math.atan2(dy, dx))
    return angle_to_quat((0, 0, heading_deg + TRACK_HEADING_ROT_OFFSET_DEG))


def _unit_tangent(a: "TrackPoint", b: "TrackPoint"):
    """Unit vector from point a to point b in the x/y plane."""
    dx, dy = b.x - a.x, b.y - a.y
    dist = math.hypot(dx, dy) or 1.0
    return dx / dist, dy / dist


def _signed_line_progress(pos_xy, line_point, line_tangent):
    """Signed distance of pos_xy along line_tangent, measured from
    line_point - negative "behind" the line, >=0 "at or past" it. A sign
    flip between two consecutive samples (negative -> >=0) means the
    infinite line perpendicular to line_tangent through line_point was
    crossed moving forward. See _lateral_offset_from_line() for bounding
    this to a finite gate."""
    return (pos_xy[0] - line_point[0]) * line_tangent[0] + (pos_xy[1] - line_point[1]) * line_tangent[1]


def _lateral_offset_from_line(pos_xy, line_point, line_tangent):
    """Perpendicular distance of pos_xy from line_point, measured along the
    normal of line_tangent. Used to bound a crossing to the finite gate
    around the start/finish point (FINISH_LINE_HALF_WIDTH_M) rather than
    the infinite line _signed_line_progress() checks the sign of."""
    nx, ny = -line_tangent[1], line_tangent[0]
    return (pos_xy[0] - line_point[0]) * nx + (pos_xy[1] - line_point[1]) * ny


def closest_point_on_polyline(px, py, track: TrackDefinition, prev_along=None,
                               search_window_m=None, precomputed=None):
    """Cross-track error helper (TRACK_CONE_AI_INTEGRATION.md §5): projects
    (px, py) onto the track centerline. Returns (distance_along_track_m,
    signed_lateral_offset_m, total_track_length_m). Signed offset is
    positive to the left of direction of travel, negative to the right
    (consistent with the left-hand normal used in
    compute_boundary_offsets() above).

    By default (prev_along=None) this searches every segment on the track
    and returns the single globally closest one - fine for a one-off call,
    but NOT what a per-poll telemetry loop should do on a track that
    self-intersects (see TRACK_PROGRESS_SEARCH_WINDOW_M above): two
    segments can sit at nearly the same (x, y) while being far apart in
    arc length, and a fresh global search can't tell them apart.

    Pass prev_along (the distance_along_track_m result from the previous
    call) together with search_window_m to instead restrict the search to
    segments within that many arc-length meters of prev_along, so progress
    is tracked locally rather than re-discovered from scratch every poll.
    Falls back to a global search if that window happens to contain no
    segments (shouldn't normally happen, but a track edit or a very small
    window shouldn't be able to produce a crash).

    precomputed, if given, should be the (pts, cumulative) tuple already
    returned by _cumulative_arc_length(track.points, track.closed_loop) -
    callers that invoke this every poll tick (e.g.
    run_beamng_track_test()'s telemetry loop) should compute that once
    outside the loop and pass it in here, since the track itself never
    changes mid-run."""
    pts, cumulative = precomputed if precomputed is not None else \
        _cumulative_arc_length(track.points, track.closed_loop)
    total = cumulative[-1]

    if prev_along is not None and search_window_m is not None:
        search_indices = _segment_indices_near(
            cumulative, total, track.closed_loop, prev_along, search_window_m
        )
        if not search_indices:
            search_indices = range(len(pts) - 1)
    else:
        search_indices = range(len(pts) - 1)

    best_dist_sq = None
    best_along = 0.0
    best_lateral = 0.0
    for i in search_indices:
        a, b = pts[i], pts[i + 1]
        abx, aby = b.x - a.x, b.y - a.y
        seg_len_sq = abx * abx + aby * aby
        if seg_len_sq < 1e-9:
            t = 0.0
        else:
            t = ((px - a.x) * abx + (py - a.y) * aby) / seg_len_sq
            t = max(0.0, min(1.0, t))
        proj_x, proj_y = a.x + t * abx, a.y + t * aby
        dx, dy = px - proj_x, py - proj_y
        dist_sq = dx * dx + dy * dy
        if best_dist_sq is None or dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_along = cumulative[i] + t * math.hypot(abx, aby)
            seg_len = math.hypot(abx, aby)
            if seg_len < 1e-9:
                best_lateral = 0.0
            else:
                # Left-hand normal of the segment direction, matching
                # compute_boundary_offsets()'s convention above.
                nx, ny = -aby / seg_len, abx / seg_len
                best_lateral = dx * nx + dy * ny
    return best_along, best_lateral, total


def _cumulative_arc_length(points, closed_loop):
    """Returns a list of cumulative distances (same length as the point
    list, or +1 if closed_loop, for the synthetic closing segment) and the
    total track length."""
    pts = list(points) + ([points[0]] if closed_loop else [])
    cumulative = [0.0]
    for prev, cur in zip(pts, pts[1:]):
        cumulative.append(cumulative[-1] + math.hypot(cur.x - prev.x, cur.y - prev.y))
    return pts, cumulative


def _segment_indices_near(cumulative, total, closed_loop, prev_along, window_m):
    """Returns the indices i of segments (pts[i] -> pts[i+1]) whose
    arc-length span is within window_m of prev_along - the along-track
    distance from the *previous* poll, not physical (x, y) distance. This
    is what lets closest_point_on_polyline() distinguish two segments that
    happen to sit at/near the same physical location (a self-intersecting
    track, e.g. a figure-8) but are far apart in arc length: only the one
    near where the car actually was last is considered.

    For closed loops, distance wraps around the seam (total <-> 0), e.g.
    prev_along near `total` should still consider segments near 0."""
    n_segs = len(cumulative) - 1

    def _arc_dist(d):
        raw = abs(d - prev_along)
        return min(raw, total - raw) if closed_loop else raw

    return [
        i for i in range(n_segs)
        if min(_arc_dist(cumulative[i]), _arc_dist(cumulative[i + 1])) <= window_m
    ]


# ==========================================
# 2. BEAMNG AUTOMATION BRIDGE
# ==========================================
# Entry points first (load_beamng_scenario, run_beamng_test - what the
# Streamlit UI calls directly), then the camera/native-start helpers used
# internally by run_beamng_test's "Run AI Lap"-style flow, ordered so
# _run_native_start_and_test (the one that kicks the others off) appears
# above wait_for_native_start, which appears above resume_and_focus_camera,
# which appears above set_camera_mode - each one only below the thing that
# calls it.
def load_beamng_scenario(bng_home=None):
    """Launch BeamNG, build/load the scenario, and spawn the test vehicle -
    but stop short of driving or starting the timer. This is the "scenario
    start menu" step: it lets the level finish loading (and lets the user
    visually confirm the car spawned correctly) before the timed test run
    begins, so level/game load time no longer bleeds into the average-speed
    and telemetry numbers. See run_beamng_test() for the actual timed run,
    which is kicked off separately once the user presses Start."""
    if not BEAMNG_AVAILABLE:
        return {"error": "beamngpy library not installed or configured."}
    
    if not bng_home or not validate_beamng_directory(bng_home):
        return {"error": "Invalid or missing BeamNG home folder."}

    # If an earlier test is still sitting open/paused (see the pause-instead-
    # of-close change below), close it out before launching a fresh instance
    # so we don't leak connections or collide on the game's network port.
    if st.session_state.get('active_bng_client') is not None:
        try:
            st.session_state.active_bng_client.close()
        except Exception:
            pass
        st.session_state.active_bng_client = None
        st.session_state.active_bng_vehicle = None

    try:
        beamng_client = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT, home=bng_home)
        beamng_client.open(launch=True)

        # Unique naming to avoid prefab collision caches
        unique_suffix = int(time.time())
        scenario_name = f'ai_suspension_validation_{unique_suffix}'

        # Safely resolve user folder path if available to prevent NoneType crashes
        user_path_str = getattr(beamng_client, 'user', None)
        if user_path_str:
            user_folder = Path(user_path_str)
            for subpath in [f"levels/{BEAMNG_MAP}/scenarios/{scenario_name}.json", f"levels/{BEAMNG_MAP}/main/MissionGroup/ai_suspension_validation*"]:
                for p in user_folder.glob(subpath):
                    try:
                        if p.is_file():
                            p.unlink()
                    except Exception:
                        pass

        scenario = Scenario(BEAMNG_MAP, scenario_name)
        test_vehicle = Vehicle('ego_vehicle', model='etk800', license=BEAMNG_VEHICLE_LICENSE)
        # NOTE: we deliberately do NOT call scenario.add_vehicle() here.
        # On this BeamNG build, prefab-embedded vehicles are failing to
        # actually spawn into the world (see the "Player vehicle not found"
        # log line, and the later Lua "attempt to index local 'veh' (a nil
        # value)" crash in techCore.lua when beamngpy tries to connect to
        # it) — almost certainly a prefab-schema mismatch between this game
        # build and the installed beamngpy version's prefab.json template.
        # Instead we load an empty scenario, then spawn the vehicle live at
        # runtime via VehiclesApi.spawn(), which uses a completely separate
        # "SpawnVehicle" message path that doesn't depend on the prefab file
        # at all.

        # Build scenario files in user folder
        scenario.make(beamng_client)

        # Verify BeamNGpy actually finished writing both files before we ask
        # the game to load them. If the .prefab is missing, scenario.make()
        # partially failed (often due to a beamngpy/game-version mismatch),
        # and proceeding here is exactly what leaves the game stuck on the
        # "ai_suspension_validation_..." start screen with no vehicle spawned.
        if user_path_str:
            expected_prefab = Path(user_path_str) / "levels" / BEAMNG_MAP / "scenarios" / f"{scenario_name}.prefab"
            expected_json = Path(user_path_str) / "levels" / BEAMNG_MAP / "scenarios" / f"{scenario_name}.json"
            if not expected_prefab.exists() or not expected_json.exists():
                beamng_client.close()
                return {
                    "error": (
                        f"scenario.make() did not produce both scenario files. "
                        f"json exists={expected_json.exists()} ({expected_json}), "
                        f"prefab exists={expected_prefab.exists()} ({expected_prefab}). "
                        f"This usually means beamngpy=={beamngpy.__version__} does not match "
                        f"this BeamNG build's scenario format. Check for a beamngpy update, "
                        f"or downgrade the game to a version your beamngpy release supports."
                    )
                }

        # Official scenario loading flow (loads an empty level - no vehicle yet)
        beamng_client.scenario.load(scenario)
        beamng_client.scenario.start()

        # Freeze the sim right here. scenario.start() is what puts up
        # BeamNG's own "Start" screen (paused, free camera, orange Start
        # button) - pausing now means the game actually sits frozen on that
        # screen instead of quietly continuing to load/spawn/move things
        # underneath it while it's shown. Everything below (spawning the
        # vehicle, attaching sensors, setting the camera) happens while
        # still paused; we only resume once the player is genuinely meant
        # to take over - see wait_for_native_start(), which lifts this
        # pause right before it starts polling for the accelerator press.
        try:
            beamng_client.pause()
        except Exception:
            pass

        # Allow physics engine and camera viewport to initialize
        time.sleep(3.0)

        # Spawn the vehicle live, at runtime, bypassing the broken prefab path.
        # spawn() also calls vehicle.connect() internally when connect=True,
        # so test_vehicle is fully usable immediately after this returns.
        # (Spawning is a scene-graph operation, not a physics step, so this
        # works fine while the sim is paused.)
        #
        # NOTE: because there's no vehicle in the prefab, BeamNG's own
        # scenario system logs a "Player vehicle not found" warning and its
        # native "Start" button on the paused screen won't do anything
        # useful if clicked (it's looking for a prefab vehicle that was
        # never there by design - see the note above run_beamng_test()).
        # That's expected and harmless here: don't click it. Once the car
        # is visible in the world below, just press the accelerator.
        spawn_success = beamng_client.vehicles.spawn(
            test_vehicle, pos=BEAMNG_SPAWN_POS, rot_quat=BEAMNG_SPAWN_ROT_QUAT, connect=True
        )
        # Force the simulator focus and camera to jump onto the vehicle
        test_vehicle.switch()
        if not spawn_success:
            beamng_client.close()
            return {"error": "beamng_client.vehicles.spawn() reported failure - vehicle did not spawn."}

        # WORKAROUND: on this BeamNG build, vehicles.spawn()'s pos argument
        # has been observed to be silently ignored while the sim is paused
        # (see the pause() call above) - the vehicle materializes at the
        # level's own default/center spawnpoint on gridmap_v2 instead of
        # BEAMNG_SPAWN_POS. teleport() goes over a separate message path
        # than spawn(), so issuing it immediately afterward forces the
        # position regardless of whether spawn's own pos was honored.
        try:
            beamng_client.vehicles.teleport(
                test_vehicle,
                pos=(BEAMNG_SPAWN_POS[0], BEAMNG_SPAWN_POS[1], BEAMNG_SPAWN_POS[2] + TELEPORT_Z_CLEARANCE_M),
                rot_quat=BEAMNG_SPAWN_ROT_QUAT, reset=True,
            )
        except Exception as teleport_exc:
            beamng_client.close()
            return {"error": f"Post-spawn teleport to BEAMNG_SPAWN_POS failed: {teleport_exc}"}

        # Verify where the vehicle actually ended up instead of trusting
        # spawn_success/teleport's return value blindly - if this still
        # doesn't match BEAMNG_SPAWN_POS, the mismatch will be visible in
        # the returned dict instead of only showing up as "car's in a wall"
        # after the fact.
        actual_spawn_pos = None
        try:
            test_vehicle.sensors.poll()
            state_data = getattr(test_vehicle, 'state', None)
            if state_data and isinstance(state_data, dict) and 'pos' in state_data:
                actual_spawn_pos = tuple(state_data['pos'])
        except Exception as poll_exc:
            st.warning(f"⚠️ Could not verify spawn position after teleport ({poll_exc}) - proceeding unverified.")
        if actual_spawn_pos is None:
            st.warning(
                "⚠️ Spawn position could not be read back from the vehicle's "
                "state sensor - the teleport may or may not have taken "
                "effect; this run is proceeding without that check."
            )
        else:
            drift = math.dist(actual_spawn_pos[:2], BEAMNG_SPAWN_POS[:2])
            if drift > 5.0:
                beamng_client.close()
                return {
                    "error": (
                        f"Vehicle spawned at {actual_spawn_pos}, "
                        f"{drift:.1f}m from the intended BEAMNG_SPAWN_POS "
                        f"{BEAMNG_SPAWN_POS}. Spawn/teleport did not take "
                        f"effect - check the BeamNG/beamngpy version match."
                    )
                }
            else:
                st.info(f"✅ Vehicle confirmed at {actual_spawn_pos} (target {BEAMNG_SPAWN_POS}, drift {drift:.2f}m).")

        # Put the player's camera/controls into the vehicle. Without this,
        # the spawned vehicle just sits there with nobody "seated" in it,
        # which is why free-cam was the only view available before.
        beamng_client.vehicles.switch(test_vehicle)

        # Free/non-tech sensor - exposes wheel-speed data (wheelspeed per
        # wheel isn't guaranteed on every vehicle/build, but this pulls in
        # whatever real per-wheel signals the electrics system publishes).
        try:
            test_vehicle.sensors.attach('electrics', Electrics())
        except Exception:
            pass

        # NOTE: switch() only makes this the "active" vehicle for input/dash
        # purposes - it does not by itself pull the game out of Free Camera
        # mode. Camera mode is a separate, per-vehicle setting. Query the
        # modes this build actually supports and request an in-car view,
        # falling back gracefully if none of our preferred names exist.
        try:
            available_camera_modes = beamng_client.camera.get_player_modes(test_vehicle)
            preferred_modes = ["onboard", "driver", "hood", "chase", "orbit"]
            chosen_mode = next(
                (m for m in preferred_modes if m in available_camera_modes),
                None,
            )
            if chosen_mode:
                beamng_client.camera.set_player_mode(test_vehicle, chosen_mode, config={})
        except Exception:
            # Non-critical: if camera mode switching fails, the validation
            # run itself should still proceed and produce a result.
            pass

        # AI driving is disabled up front (raw throttle/steering is used
        # during the actual test - see run_beamng_test()), and the vehicle
        # is held stationary with the brake here. It just sits ready-and-
        # loaded in the level until the user presses the accelerator inside
        # BeamNG itself (see wait_for_native_start()).
        test_vehicle.ai.set_mode('disabled')
        test_vehicle.control(throttle=0.0, steering=0.0, brake=1.0)

        return {"status": "Ready", "client": beamng_client, "vehicle": test_vehicle}

    except Exception as e:
        import traceback
        # Don't leave a half-initialized connection dangling on error.
        if 'beamng_client' in locals():
            try:
                beamng_client.close()
            except Exception:
                pass
        return {"error": str(e), "traceback": traceback.format_exc()}


def run_beamng_test(beamng_client, test_vehicle):
    """Drive the already-loaded/spawned vehicle through the timed test run
    and collect telemetry. Call load_beamng_scenario() first and pass its
    returned client/vehicle in here - the timer and average-speed window
    both start at the top of this function, not at scenario load, which is
    the whole point of separating the two steps."""
    try:
        # Drive the vehicle forward directly via raw throttle/steering input,
        # instead of BeamNG's built-in AI driver. ai.set_mode('speed') only
        # works when the vehicle is on a road BeamNG's navgraph recognizes -
        # the custom spawn point captured from the World Editor isn't
        # guaranteed to be on one, which is why the car was just sitting
        # there stationary. Sending throttle directly guarantees forward
        # motion regardless of road/nav data.
        #
        # Constant throttle=1.0 has no notion of a target speed, so the car
        # just kept accelerating for the whole run. Instead we hold speed
        # with a simple proportional controller: throttle/brake are nudged
        # each poll based on how far current speed is from BEAMNG_AI_SPEED.
        SPEED_CONTROL_GAIN = 0.15

        def _speed_control(current_speed):
            speed_error = BEAMNG_AI_SPEED - current_speed
            if speed_error > 0:
                throttle = min(1.0, max(0.05, speed_error * SPEED_CONTROL_GAIN))
                return throttle, 0.0
            else:
                brake = min(1.0, max(0.0, -speed_error * SPEED_CONTROL_GAIN))
                return 0.0, brake

        # Initial nudge to get it rolling before the first speed reading comes in.
        test_vehicle.control(throttle=0.6, steering=0.0, brake=0.0)

        # AdvancedIMU is a BeamNG.tech-only sensor - it's rejected server-side
        # with "This feature requires a BeamNG.tech license" on plain
        # BeamNG.drive (which is what's installed here). The basic `state`
        # sensor has no acceleration field either. So instead of a dedicated
        # accel sensor, we finite-difference world-space vertical velocity
        # (state['vel'][2]) between consecutive polls to get vertical
        # acceleration - this needs no special license, and it's actually a
        # closer match to the Python model's own world-vertical "chassis
        # heave" definition than a vehicle-local IMU reading would be.
        start_time = time.time()
        vertical_g_forces = []
        previous_vel_z = None
        previous_sample_time = None

        # Full telemetry log - one entry appended per successful poll, so
        # the UI can show more than just the single peak-G number.
        telemetry = {
            "elapsed_s": [],
            "vertical_g": [],
            "speed_mps": [],
            "pos_x": [],
            "pos_y": [],
            "pos_z": [],
            "throttle": [],
            "brake": [],
            "avg_wheel_av": [],  # real per-wheel-derived signal from the Electrics sensor, where available
        }

        # Live view: a placeholder redrawn periodically during the loop so
        # you can watch the run instead of only seeing results at the end.
        # Redrawing a matplotlib figure on every single 20ms poll (50Hz)
        # would badly bottleneck the control loop, so redraws are throttled
        # to a fixed wall-clock interval instead of every sample.
        live_chart_placeholder = st.empty()
        LIVE_CHART_UPDATE_INTERVAL_S = 0.25
        last_chart_update_time = 0.0

        # If BeamNG stops responding for this many consecutive polls
        # (roughly DISCONNECT_FAILURE_LIMIT * BEAMNG_POLL_INTERVAL seconds -
        # about 1s by default), treat it as the game having been closed
        # manually mid-run rather than silently finishing out the loop with
        # no new data and reporting a bogus "Success".
        DISCONNECT_FAILURE_LIMIT = 50
        consecutive_failures = 0
        connection_lost = False

        while time.time() - start_time < BEAMNG_TEST_DURATION:
            try:
                test_vehicle.sensors.poll()
                state_data = getattr(test_vehicle, 'state', None)
                if state_data and isinstance(state_data, dict) and 'vel' in state_data:
                    current_time = state_data.get('time', time.time())
                    current_vel_z = state_data['vel'][2]

                    # Hold speed using measured horizontal velocity (x, y);
                    # vertical (z) is excluded since that's the bump motion
                    # we're trying to measure, not forward progress.
                    horizontal_speed = (state_data['vel'][0] ** 2 + state_data['vel'][1] ** 2) ** 0.5
                    throttle_cmd, brake_cmd = _speed_control(horizontal_speed)
                    test_vehicle.control(throttle=throttle_cmd, steering=0.0, brake=brake_cmd)

                    sample_vertical_g = 0.0
                    if previous_vel_z is not None and previous_sample_time is not None:
                        dt = current_time - previous_sample_time
                        if dt > 1e-6:
                            vertical_accel = (current_vel_z - previous_vel_z) / dt
                            sample_vertical_g = abs(vertical_accel) / GRAVITY
                            vertical_g_forces.append(sample_vertical_g)

                    previous_vel_z = current_vel_z
                    previous_sample_time = current_time

                    vehicle_pos = state_data.get('pos', (None, None, None))
                    electrics_data = getattr(test_vehicle, 'electrics', None) or {}
                    avg_wheel_av = electrics_data.get('avg_wheel_av') if isinstance(electrics_data, dict) else None

                    telemetry["elapsed_s"].append(time.time() - start_time)
                    telemetry["vertical_g"].append(sample_vertical_g)
                    telemetry["speed_mps"].append(horizontal_speed)
                    telemetry["pos_x"].append(vehicle_pos[0])
                    telemetry["pos_y"].append(vehicle_pos[1])
                    telemetry["pos_z"].append(vehicle_pos[2])
                    telemetry["throttle"].append(throttle_cmd)
                    telemetry["brake"].append(brake_cmd)
                    telemetry["avg_wheel_av"].append(avg_wheel_av)

                    now = time.time()
                    if now - last_chart_update_time >= LIVE_CHART_UPDATE_INTERVAL_S:
                        last_chart_update_time = now
                        live_fig, live_height_ax = plt.subplots(figsize=(8, 3))
                        live_height_ax.plot(telemetry["elapsed_s"], telemetry["pos_z"], color="royalblue", linewidth=1.3)
                        live_height_ax.set_xlabel("Time (s)")
                        live_height_ax.set_ylabel("Height (m)")
                        live_height_ax.set_title(
                            f"Live: Chassis Height (Up/Down) - "
                            f"{telemetry['elapsed_s'][-1]:.1f}s / {BEAMNG_TEST_DURATION}s"
                        )
                        live_height_ax.grid(True, linestyle="--", alpha=0.6)
                        live_fig.tight_layout()
                        live_chart_placeholder.pyplot(live_fig)
                        plt.close(live_fig)
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= DISCONNECT_FAILURE_LIMIT:
                    connection_lost = True
                    break

            time.sleep(BEAMNG_POLL_INTERVAL)

        live_chart_placeholder.empty()

        if connection_lost:
            # The game vanished mid-run (e.g. closed manually) - the client
            # object is no longer good for anything, so don't try to pause()
            # through it or hand it back to the UI as if the session were
            # still open. Report this plainly instead of returning a
            # "Success" built from a truncated/stale telemetry log.
            return {
                "error": (
                    "Lost connection to BeamNG during the test - it looks like the "
                    "game was closed or crashed mid-run. Partial telemetry "
                    f"({len(telemetry['elapsed_s'])} samples over "
                    f"{telemetry['elapsed_s'][-1]:.1f}s) was collected before the "
                    "connection dropped, but the run wasn't completed."
                ),
                "connection_lost": True,
                "partial_telemetry": telemetry,
            }

        # Bring the car to a stop, then pause the sim and leave the
        # connection open (rather than closing it) so the user can inspect
        # the result in-game. The client/vehicle are stashed in session
        # state; a "Terminate BeamNG Session" button in the UI closes it
        # explicitly when the user is done.
        test_vehicle.control(throttle=0.0, brake=1.0)
        try:
            beamng_client.pause()
        except Exception:
            pass

        st.session_state.active_bng_client = beamng_client
        st.session_state.active_bng_vehicle = test_vehicle

        peak_vertical_g = max(vertical_g_forces) if vertical_g_forces else 1.25
        avg_speed = sum(telemetry["speed_mps"]) / len(telemetry["speed_mps"]) if telemetry["speed_mps"] else 0.0
        distance_traveled = 0.0
        xs, ys = telemetry["pos_x"], telemetry["pos_y"]
        for i in range(1, len(xs)):
            if xs[i] is not None and xs[i - 1] is not None:
                distance_traveled += ((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2) ** 0.5

        return {
            "status": "Success",
            "peak_vertical_g": peak_vertical_g,
            "avg_speed_mps": avg_speed,
            "distance_traveled_m": distance_traveled,
            "sample_count": len(telemetry["elapsed_s"]),
            "test_duration_s": BEAMNG_TEST_DURATION,
            "telemetry": telemetry,
        }

    except Exception as e:
        import traceback
        # Unlike load_beamng_scenario(), we deliberately do NOT close
        # beamng_client here on error - the scenario is already loaded and
        # the connection is stashed in session state, so the user can still
        # inspect what happened in-game and use "Terminate BeamNG Session"
        # to close it explicitly, rather than losing the loaded level.
        return {"error": str(e), "traceback": traceback.format_exc()}


def _run_native_start_and_test(client, vehicle):
    """Shared helper: wait for the player to press the accelerator inside
    BeamNG, then run the timed test if they did. Used by both the initial
    "Load Scenario" click and the "Resume Waiting" fallback so disconnect/
    timeout handling only has to live in one place."""
    outcome = wait_for_native_start(client, vehicle)

    if outcome == "disconnected":
        st.session_state.last_bng_result = {
            "error": (
                "Lost connection to BeamNG while waiting for you to start the "
                "test - it looks like the game was closed or crashed before "
                "the accelerator was pressed."
            ),
            "connection_lost": True,
        }
        st.session_state.active_bng_client = None
        st.session_state.active_bng_vehicle = None
        st.session_state.bng_scenario_loaded = False
        return

    if outcome == "timeout":
        # Scenario stays loaded and ready - nothing to do here.
        return

    # outcome == "started"
    with st.spinner("Running timed AI test drive..."):
        validation_result = run_beamng_test(client, vehicle)
    st.session_state.last_bng_result = validation_result
    st.session_state.bng_scenario_loaded = False
    if validation_result.get("connection_lost"):
        # The client is dead - don't leave it in session state as if the
        # game were still sitting open and pausable.
        st.session_state.active_bng_client = None
        st.session_state.active_bng_vehicle = None


def wait_for_native_start(beamng_client, test_vehicle, poll_interval=0.05,
                           throttle_threshold=0.08, speed_threshold_mps=0.3,
                           timeout_s=None, disconnect_failure_limit=40):
    """Block until the player triggers the start from *inside BeamNG itself*
    - by pressing the accelerator - rather than from a button on the
    Streamlit site. This is the "native start menu" step: once the level is
    loaded (see load_beamng_scenario()), the only remaining action needed
    to kick off the timed run happens in-game, not on the website.

    Lifts the pause load_beamng_scenario() put the sim into, so the player
    can actually move the car once they're ready.

    Polls two independent signals each tick so this doesn't depend on one
    exact Electrics field name matching this BeamNG build:
      1. The Electrics sensor's raw pedal input (throttle_input, falling
         back to throttle if that key isn't present on this build).
      2. Actual chassis speed picked up from the free `state` sensor, in
         case the input field is named/reported differently - if the car
         is already moving under its own power, that counts as "started"
         too.

    Returns "started" once the accelerator is detected, "timeout" if
    timeout_s elapses first (only relevant if a timeout is supplied - the
    default is to wait indefinitely), or "disconnected" if BeamNG stops
    responding for disconnect_failure_limit consecutive polls in a row -
    e.g. the game window was closed manually while we were waiting.
    """
    # NOTE: this used to be a single bare `except Exception: pass` around
    # the whole resume+camera block, which is why the free-cam issue was
    # invisible for so long. Now shared via resume_and_focus_camera() so
    # every flow (including the track-run flow) surfaces failures the same
    # way instead of each one swallowing them separately.
    camera_warning = resume_and_focus_camera(beamng_client, test_vehicle)

    if camera_warning:
        st.warning(
            f"⚠️ Camera may still be in free-cam: {camera_warning} "
            "You can switch views manually in-game (default key is usually "
            "F7/Numpad keys depending on your BeamNG bindings)."
        )

    status_placeholder = st.empty()
    waited = 0.0
    consecutive_failures = 0
    try:
        while True:
            try:
                test_vehicle.sensors.poll()
                electrics_data = getattr(test_vehicle, 'electrics', None) or {}
                raw_throttle = electrics_data.get('throttle_input', electrics_data.get('throttle', 0.0)) or 0.0

                state_data = getattr(test_vehicle, 'state', None)
                current_speed = 0.0
                if state_data and isinstance(state_data, dict) and 'vel' in state_data:
                    vx, vy = state_data['vel'][0], state_data['vel'][1]
                    current_speed = (vx ** 2 + vy ** 2) ** 0.5
                consecutive_failures = 0
            except Exception:
                raw_throttle, current_speed = 0.0, 0.0
                consecutive_failures += 1
                if consecutive_failures >= disconnect_failure_limit:
                    status_placeholder.empty()
                    return "disconnected"

            status_placeholder.info(
                "🎮 Scenario loaded - press the accelerator in BeamNG to start the "
                f"timed test. ({waited:.0f}s waiting)"
            )

            if raw_throttle >= throttle_threshold or current_speed >= speed_threshold_mps:
                status_placeholder.empty()
                return "started"

            if timeout_s is not None and waited >= timeout_s:
                status_placeholder.empty()
                return "timeout"

            time.sleep(poll_interval)
            waited += poll_interval
    finally:
        status_placeholder.empty()


def resume_and_focus_camera(beamng_client, test_vehicle,
                             preferred_modes=("onboard", "driver", "hood", "chase", "orbit")):
    """Resume the sim (if paused) and switch the player camera onto
    test_vehicle, returning a human-readable warning string on any failure
    or None on success. Shared by every flow that needs the camera to
    actually follow the car once it's actually driving - previously each
    flow duplicated this logic with its own bare `except Exception: pass`,
    which is why camera/resume failures kept going unnoticed. Any caller
    should st.warning() a non-None return value rather than swallow it
    again."""
    try:
        beamng_client.resume()
    except Exception as resume_exc:
        return f"beamng_client.resume() failed: {resume_exc}"
    return set_camera_mode(beamng_client, test_vehicle, preferred_modes)


def set_camera_mode(beamng_client, test_vehicle, preferred_modes):
    """Switches to test_vehicle and puts the player camera into the first
    available mode from preferred_modes (checked in order), returning a
    human-readable warning string on any failure or None on success.

    Pulled out of resume_and_focus_camera() so a caller that wants the
    camera to land on the vehicle right away - e.g. immediately after it
    spawns, while the sim is still deliberately paused for setup (cones,
    sensors, etc.) - doesn't also have to resume the sim just to move the
    camera. resume_and_focus_camera() below is now a thin wrapper around
    this for callers that DO want both."""
    try:
        beamng_client.vehicles.switch(test_vehicle)
        available_camera_modes = beamng_client.camera.get_player_modes(test_vehicle)
        chosen_mode = next(
            (m for m in preferred_modes if m in available_camera_modes),
            None,
        )
        if chosen_mode:
            beamng_client.camera.set_player_mode(test_vehicle, chosen_mode, config={})
            return None
        return f"No preferred camera mode found in available modes: {available_camera_modes!r}"
    except Exception as cam_exc:
        return f"Camera mode switch failed: {cam_exc}"


# ==========================================
# 2.5 TRACK / CONE AI-DRIVING BEAMNG BRIDGE
# ==========================================
# Same "empty scenario + live spawn" pattern as load_beamng_scenario() above
# (see that function's comments for why), extended to also place cones and
# drive the track with vehicle.ai.execute_script() instead of raw
# throttle/steering. See TRACK_CONE_AI_INTEGRATION.md for the full design
# writeup - this is a direct implementation of that spec, and the two files
# should be read together.
# load_beamng_track_scenario() (build/load the scenario) comes before
# run_beamng_track_test() (drive it and collect telemetry) - the two don't
# call each other, so this is simply the order the Streamlit UI calls them
# in: load the scenario first, then run the test.

def load_beamng_track_scenario(bng_home, track: TrackDefinition, cone_spacing_m=DEFAULT_CONE_SPACING_M):
    """Launch BeamNG, load an empty scenario, spawn the test vehicle at the
    track's first point, then live-spawn a cone at every position in
    generate_all_cone_positions()'s output (see that function for how the
    track-boundary and run-up-lane cone sets are combined/filtered).
    Mirrors load_beamng_scenario()'s structure closely on purpose.

    Cones are spawned as live Vehicle instances (model=BEAMNG_CONE_VEHICLE_MODEL,
    part_config=BEAMNG_CONE_PROP_MODEL) after the ego vehicle is confirmed in
    place, not baked into the scenario prefab - see the comments in this
    function's body for why the prefab route doesn't work on this build.

    Returns a dict with "client", "vehicle", "cone_positions", and
    "cones_requested"/"cones_confirmed" counts on success, or "error" on
    failure. Cone placement is best-effort past the point the vehicle is
    spawned and drivable - individual cone spawn failures don't abort the
    scenario, only show up as a lower "cones_confirmed" count and a
    "cone_warning" key, since a track with some cones missing is still
    useful to drive and inspect. Likewise, a camera-focus failure shows up
    as a "camera_warning" key rather than aborting the scenario - the run
    is still perfectly usable from a free/default camera.
    """
    if not BEAMNG_AVAILABLE:
        return {"error": "beamngpy library not installed or configured."}
    if not bng_home or not validate_beamng_directory(bng_home):
        return {"error": "Invalid or missing BeamNG home folder."}
    if not BEAMNG_CONE_PROP_MODEL:
        return {
            "error": (
                "BEAMNG_CONE_PROP_MODEL is not set. Open the in-game vehicle/"
                "prop selector, find the exact internal name of a cone/pylon "
                "prop for your installed BeamNG version, and set "
                "BEAMNG_CONE_PROP_MODEL to it near the other BEAMNG_* "
                "constants (see TRACK_CONE_AI_INTEGRATION.md §0/§3 - this is "
                "deliberately not guessed)."
            )
        }

    if st.session_state.get('track_bng_client') is not None:
        try:
            st.session_state.track_bng_client.close()
        except Exception:
            pass
        st.session_state.track_bng_client = None
        st.session_state.track_bng_vehicle = None

    cone_positions = generate_all_cone_positions(track, cone_spacing_m=cone_spacing_m, run_up_m=RUN_UP_DISTANCE_M)
    # Spawn RUN_UP_DISTANCE_M behind the track's first point, facing along
    # its initial tangent, instead of exactly on node 0 using the fixed
    # BEAMNG_SPAWN_ROT_QUAT - see the comment on RUN_UP_DISTANCE_M above for
    # why (this is what was causing the AI to back up/spin in place at the
    # start of every track run).
    run_up_pos, spawn_rot_quat = compute_run_up_start(track, run_up_m=RUN_UP_DISTANCE_M)
    start_pos = (
        run_up_pos[0], run_up_pos[1],
        run_up_pos[2] if run_up_pos[2] is not None else BEAMNG_SPAWN_POS[2],
    )

    try:
        beamng_client = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT, home=bng_home)
        beamng_client.open(launch=True)

        unique_suffix = int(time.time())
        scenario_name = f'ai_track_run_{unique_suffix}'
        scenario = Scenario(BEAMNG_MAP, scenario_name)

        # Cones are NOT added to the scenario/prefab here (no
        # scenario.add_object(StaticObject(...)) - that was the previous
        # version of this function, and it never actually worked, for two
        # independent reasons:
        #   1. Prefab-embedded objects are already known broken on this
        #      BeamNG build - see the "prefab-embedded vehicles" comment on
        #      the ego vehicle in load_beamng_scenario() above. Baking
        #      cones into the same prefab hits that identical failure mode,
        #      so they never appeared regardless of what shape/model they
        #      pointed at.
        #   2. BEAMNG_CONE_PROP_MODEL ("vehicles/cones/...") is a vehicle
        #      model + part-config path (confirmed in-game by the user),
        #      not a static-mesh shape path - StaticObject's "shape" field
        #      expects the latter, so even a working prefab-embedding story
        #      wouldn't have resolved it correctly.
        # Cones are instead spawned live, further down, the exact same way
        # as the ego vehicle (vehicles.spawn() at runtime) - see the cone
        # spawn loop below, after the ego vehicle is confirmed in place.
        test_vehicle = Vehicle('ego_vehicle', model='etk800', license=BEAMNG_VEHICLE_LICENSE)
        scenario.make(beamng_client)

        user_path_str = getattr(beamng_client, 'user', None)
        if user_path_str:
            expected_prefab = Path(user_path_str) / "levels" / BEAMNG_MAP / "scenarios" / f"{scenario_name}.prefab"
            expected_json = Path(user_path_str) / "levels" / BEAMNG_MAP / "scenarios" / f"{scenario_name}.json"
            if not expected_prefab.exists() or not expected_json.exists():
                beamng_client.close()
                return {"error": "scenario.make() did not produce both scenario files (see load_beamng_scenario() for this same check)."}

        beamng_client.scenario.load(scenario)
        beamng_client.scenario.start()
        try:
            beamng_client.pause()
        except Exception:
            pass
        time.sleep(3.0)

        spawn_success = beamng_client.vehicles.spawn(
            test_vehicle, pos=start_pos, rot_quat=spawn_rot_quat, connect=True
        )
        if not spawn_success:
            beamng_client.close()
            return {"error": "beamng_client.vehicles.spawn() reported failure - vehicle did not spawn."}

        # Same spawn()-ignores-pos-while-paused workaround as
        # load_beamng_scenario() - force the position with teleport() since
        # it uses a different message path than spawn().
        try:
            beamng_client.vehicles.teleport(
                test_vehicle, pos=start_pos, rot_quat=spawn_rot_quat, reset=True
            )
        except Exception as teleport_exc:
            beamng_client.close()
            return {"error": f"Post-spawn teleport to track start_pos failed: {teleport_exc}"}

        # Same verification as load_beamng_scenario() - confirm where the
        # vehicle actually ended up rather than trusting the teleport call's
        # return value blindly.
        actual_spawn_pos = None
        poll_failed = False
        try:
            test_vehicle.sensors.poll()
            state_data = getattr(test_vehicle, 'state', None)
            if state_data and isinstance(state_data, dict) and 'pos' in state_data:
                actual_spawn_pos = tuple(state_data['pos'])
        except Exception as poll_exc:
            poll_failed = True
            st.warning(f"⚠️ Could not verify track spawn position after teleport ({poll_exc}) - proceeding unverified.")
        if actual_spawn_pos is None and not poll_failed:
            st.warning(
                "⚠️ Track spawn position could not be read back from the "
                "vehicle's state sensor - proceeding without that check."
            )
        elif actual_spawn_pos is not None:
            drift = math.dist(actual_spawn_pos[:2], start_pos[:2])
            if drift > 5.0:
                beamng_client.close()
                return {
                    "error": (
                        f"Vehicle spawned at {actual_spawn_pos}, {drift:.1f}m "
                        f"from the intended track start_pos {start_pos}. "
                        f"Spawn/teleport did not take effect."
                    )
                }
            else:
                st.info(f"✅ Vehicle confirmed at {actual_spawn_pos} (track start {start_pos}, drift {drift:.2f}m).")

        beamng_client.vehicles.switch(test_vehicle)

        # Free/non-tech sensors - see TRACK_CONE_AI_INTEGRATION.md §0/§5 for
        # why each of these is expected to work on plain BeamNG.drive.
        for sensor_name, sensor_obj in (
            ('electrics', lambda: Electrics()),
        ):
            try:
                test_vehicle.sensors.attach(sensor_name, sensor_obj())
            except Exception:
                pass
        try:
            from beamngpy.sensors import Timer as _BngTimer
            test_vehicle.sensors.attach('timer', _BngTimer())
        except Exception:
            pass
        try:
            from beamngpy.sensors import GForces as _BngGForces
            test_vehicle.sensors.attach('gforces', _BngGForces())
        except Exception:
            pass
        try:
            from beamngpy.sensors import Damage as _BngDamage
            test_vehicle.sensors.attach('damage', _BngDamage())
        except Exception:
            pass

        # Cones, live-spawned - see the comment above (near scenario_name)
        # for why this replaced the old prefab-embedded StaticObject
        # approach. Each cone is its own Vehicle instance
        # (model=BEAMNG_CONE_VEHICLE_MODEL, part_config=BEAMNG_CONE_PROP_MODEL,
        # the exact model/part-config pair the user confirmed in-game),
        # spawned with connect=False since a decorative cone needs no
        # sensors or control - only to physically exist in the world -
        # and opening a full sensor connection for every one of what can
        # be 100+ cones would be pure overhead.
        #
        # The "let the game snap each cone to the ground" flag is passed
        # under whichever name THIS installed beamngpy version actually
        # accepts, checked once via inspect.signature() rather than
        # hardcoded - beamngpy 1.36 renamed VehiclesApi.spawn()'s old
        # cling: bool to safe_spawn: bool (keeping cling as a deprecated
        # alias), but not every installed version necessarily has the new
        # name yet (confirmed in-game: passing safe_spawn on such a build
        # raises "unexpected keyword argument 'safe_spawn'" immediately -
        # every single cone failed with that same error, not a real
        # spawn/physics problem). Falls back to no ground-snap kwarg at
        # all on a version with neither name, rather than failing every
        # cone spawn the same way over one parameter name.
        cone_spawn_kwargs = {}
        try:
            _spawn_params = inspect.signature(beamng_client.vehicles.spawn).parameters
            if "safe_spawn" in _spawn_params:
                cone_spawn_kwargs["safe_spawn"] = True
            elif "cling" in _spawn_params:
                cone_spawn_kwargs["cling"] = True
        except (TypeError, ValueError):
            pass

        cone_warning = None
        cones_confirmed = 0
        cone_spawn_errors = []
        for i, cone in enumerate(cone_positions):
            try:
                cone_vehicle = Vehicle(
                    f"track_cone_{i}", model=BEAMNG_CONE_VEHICLE_MODEL,
                    part_config=BEAMNG_CONE_PROP_MODEL,
                )
                cone_spawned = beamng_client.vehicles.spawn(
                    cone_vehicle, pos=(cone["x"], cone["y"], cone["z"]),
                    rot_quat=(0, 0, 0, 1), connect=False, **cone_spawn_kwargs,
                )
                if cone_spawned:
                    cones_confirmed += 1
                else:
                    cone_spawn_errors.append(f"cone {i}: vehicles.spawn() reported failure")
            except Exception as cone_exc:
                cone_spawn_errors.append(f"cone {i}: {cone_exc}")
        if cones_confirmed < len(cone_positions):
            example = f" (e.g. {cone_spawn_errors[0]})" if cone_spawn_errors else ""
            cone_warning = (
                f"Only {cones_confirmed}/{len(cone_positions)} cones spawned "
                f"successfully{example}. Proceeding with the run - see the "
                "driven-path plot to check which stretches of track ended up "
                "unmarked."
            )

        test_vehicle.ai.set_mode('disabled')
        test_vehicle.control(throttle=0.0, steering=0.0, brake=1.0)

        # Snap the camera onto the vehicle in orbit mode right here, as
        # soon as the vehicle (and cones) are actually in the world -
        # not deferred until "Run AI Lap" is pressed. This was previously
        # only ever done in resume_and_focus_camera(), which also resumes
        # the sim - that's still correct for "Run AI Lap" (the sim
        # actually needs to be running for the lap), but it meant the
        # camera stayed on whatever BeamNG's own default was for the
        # entire time between loading the scenario and starting a run.
        # set_camera_mode() (unlike resume_and_focus_camera()) doesn't
        # touch the sim's pause state, so this can happen immediately
        # without prematurely resuming physics before the user is ready.
        camera_warning = set_camera_mode(beamng_client, test_vehicle, ["orbit"])

        st.session_state.track_bng_client = beamng_client
        st.session_state.track_bng_vehicle = test_vehicle

        result = {
            "status": "Ready",
            "client": beamng_client,
            "vehicle": test_vehicle,
            "cone_positions": cone_positions,
            "cones_requested": len(cone_positions),
            "cones_confirmed": cones_confirmed,
        }
        if cone_warning:
            result["cone_warning"] = cone_warning
        if camera_warning:
            result["camera_warning"] = camera_warning
        return result

    except Exception as e:
        import traceback
        if 'beamng_client' in locals():
            try:
                beamng_client.close()
            except Exception:
                pass
        return {"error": str(e), "traceback": traceback.format_exc()}


def run_beamng_track_test(beamng_client, test_vehicle, track: TrackDefinition, ai_script,
                           cone_positions, max_duration_s=None):
    """Drive the track with BeamNG's own AI via execute_script(), polling
    every free-license sensor available (see TRACK_CONE_AI_INTEGRATION.md
    §5). Stops on, in order of precedence:
      1. Crossing the start/finish line (see compute_finish_line()) - the
         primary, intended stopping condition, for both open tracks and
         closed loops (a closed loop needs a *second* crossing of the same
         line; an open track's finish line is a separate line at the last
         point) - AND, for that second/finish crossing specifically, having
         already passed all of FINISH_CHECKPOINT_COUNT checkpoints spread
         around the rest of the track (see select_finish_checkpoints()).
         Without the checkpoint requirement, a closed loop's finish gate
         being the same physical line as its start gate means the car can
         "finish" by crossing that line twice near the very start of the
         run - e.g. wobbling back and forth while the run-up settles it
         onto the path - without having driven any of the actual lap. A
         crossing that doesn't clear the checkpoint requirement is simply
         not treated as a finish; the run keeps going and waits for a real
         one.
      2. A generous timeout (ABSOLUTE_RUN_FALLBACK_S /
         RUN_TIMEOUT_SCRIPT_MULTIPLIER above) - a safety net for a crashed/
         stuck car, or an AI that falls so far behind its script's own
         schedule it never reaches the finish, so a bad run always ends
         instead of hanging the UI indefinitely.
      3. A lost connection to BeamNG.
    The old distance-along-track completion check is kept as a secondary,
    belt-and-suspenders condition for open tracks (in case the finish-line
    geometry is ever degenerate, e.g. a track with a near-zero-length final
    segment), but the finish line is what normally ends the run.
    max_duration_s, if given, overrides the computed timeout entirely
    (mainly for tests/manual overrides); leave it None in normal use.
    """
    try:
        # Computed once here (the track doesn't change mid-run) and passed
        # into every closest_point_on_polyline() call below via
        # `precomputed=` instead of being rebuilt from scratch on every
        # poll tick.
        track_pts_cumulative = _cumulative_arc_length(track.points, track.closed_loop)
        total_length = track_pts_cumulative[1][-1]
        script_duration = ai_script[-1]["t"] if ai_script else 0.0
        if max_duration_s is None:
            max_duration_s = max(
                script_duration * RUN_TIMEOUT_SCRIPT_MULTIPLIER + RUN_TIMEOUT_SCRIPT_PADDING_S,
                ABSOLUTE_RUN_FALLBACK_S,
            )

        finish_line = compute_finish_line(track)
        crossed_start_line = False
        finish_reached = False
        prev_signed_start = None
        prev_signed_finish = None

        # See FINISH_CHECKPOINT_COUNT / select_finish_checkpoints() above -
        # a finish-line crossing only counts once every checkpoint here has
        # been passed at least once during the run.
        finish_checkpoints = select_finish_checkpoints(track, cone_positions)
        checkpoints_passed = [False] * len(finish_checkpoints)
        # Diagnostic record of finish-line crossings that were rejected for
        # not having cleared the checkpoints yet - surfaced in the returned
        # dict so a run that looks like it "took a while to actually stop"
        # is explainable rather than mysterious.
        rejected_finish_crossings = []


        # Explicitly release any brake/throttle/steering input left over
        # from the post-load "hold still" state (load_beamng_track_scenario
        # sets brake=1.0 so the car doesn't roll before a run is started) -
        # execute_script is expected to take over control anyway, but a
        # held brake at the exact moment the script starts is one plausible
        # contributor to the car fighting itself at the start line, so
        # clear it defensively right before handing off to the AI.
        try:
            test_vehicle.control(throttle=0.0, steering=0.0, brake=0.0)
        except Exception:
            pass

        test_vehicle.ai.set_script(ai_script, cling=True)

        telemetry = {
            "elapsed_s": [], "sim_time_s": [], "pos_x": [], "pos_y": [], "pos_z": [],
            "speed_mps": [], "distance_along_track_m": [], "cross_track_error_m": [],
            "lat_g": [], "long_g": [], "vert_g": [], "damage_total": [],
        }
        cone_hits = []
        last_damage_total = None
        start_time = time.time()
        DISCONNECT_FAILURE_LIMIT = 50
        consecutive_failures = 0
        connection_lost = False
        distance_along = 0.0

        live_chart_placeholder = st.empty()
        LIVE_CHART_UPDATE_INTERVAL_S = 0.25
        last_chart_update_time = 0.0

        while time.time() - start_time < max_duration_s:
            try:
                test_vehicle.sensors.poll()
                state_data = getattr(test_vehicle, 'state', None)
                if state_data and isinstance(state_data, dict) and 'pos' in state_data:
                    pos = state_data['pos']
                    vel = state_data.get('vel', (0.0, 0.0, 0.0))
                    speed = math.hypot(vel[0], vel[1])

                    # prev_along=distance_along carries the *previous*
                    # poll's progress in, so the search stays local to
                    # where the car actually was last instead of a fresh
                    # global nearest-segment search every tick - see
                    # TRACK_PROGRESS_SEARCH_WINDOW_M / closest_point_on_polyline().
                    distance_along, cross_track, _total = closest_point_on_polyline(
                        pos[0], pos[1], track,
                        prev_along=distance_along,
                        search_window_m=TRACK_PROGRESS_SEARCH_WINDOW_M,
                        precomputed=track_pts_cumulative,
                    )

                    # --- Checkpoint tracking (see FINISH_CHECKPOINT_COUNT /
                    # select_finish_checkpoints() above) - update before the
                    # crossing checks below so a checkpoint reached on this
                    # exact poll already counts toward this poll's crossing
                    # decision. ---
                    for j, checkpoint in enumerate(finish_checkpoints):
                        if not checkpoints_passed[j]:
                            d = math.hypot(checkpoint["x"] - pos[0], checkpoint["y"] - pos[1])
                            if d <= FINISH_CHECKPOINT_RADIUS_M:
                                checkpoints_passed[j] = True
                    all_checkpoints_passed = all(checkpoints_passed) if finish_checkpoints else True

                    # --- Start/finish line crossing (primary stop condition) ---
                    pos_xy = (pos[0], pos[1])
                    s_start = _signed_line_progress(pos_xy, finish_line["start_point"], finish_line["start_tangent"])
                    start_lateral_ok = abs(_lateral_offset_from_line(
                        pos_xy, finish_line["start_point"], finish_line["start_tangent"]
                    )) <= FINISH_LINE_HALF_WIDTH_M
                    if prev_signed_start is not None and prev_signed_start < 0.0 <= s_start and start_lateral_ok:
                        if finish_line["same_line"]:
                            # Closed loop: first forward crossing is just
                            # leaving the run-up and starting the lap;
                            # the second is the finish - but only once the
                            # checkpoints have actually been passed. A
                            # crossing that hasn't cleared them yet is
                            # rejected outright rather than treated as a
                            # second "start": the lap already started at
                            # the first crossing, and re-arming that state
                            # here would let the count of required
                            # crossings silently grow with each rejection.
                            if not crossed_start_line:
                                crossed_start_line = True
                            elif all_checkpoints_passed:
                                finish_reached = True
                            else:
                                rejected_finish_crossings.append({
                                    "elapsed_s": time.time() - start_time,
                                    "checkpoints_passed": sum(checkpoints_passed),
                                    "checkpoints_required": len(finish_checkpoints),
                                })
                        else:
                            crossed_start_line = True
                    prev_signed_start = s_start

                    if not finish_line["same_line"]:
                        s_finish = _signed_line_progress(pos_xy, finish_line["finish_point"], finish_line["finish_tangent"])
                        finish_lateral_ok = abs(_lateral_offset_from_line(
                            pos_xy, finish_line["finish_point"], finish_line["finish_tangent"]
                        )) <= FINISH_LINE_HALF_WIDTH_M
                        if (crossed_start_line and finish_lateral_ok and prev_signed_finish is not None
                                and prev_signed_finish < 0.0 <= s_finish):
                            if all_checkpoints_passed:
                                finish_reached = True
                            else:
                                rejected_finish_crossings.append({
                                    "elapsed_s": time.time() - start_time,
                                    "checkpoints_passed": sum(checkpoints_passed),
                                    "checkpoints_required": len(finish_checkpoints),
                                })
                        prev_signed_finish = s_finish
                    # --- end start/finish line check ---

                    gforces_data = getattr(test_vehicle, 'gforces', None) or {}
                    lat_g = gforces_data.get('gx', gforces_data.get('gx2', 0.0)) if isinstance(gforces_data, dict) else 0.0
                    long_g = gforces_data.get('gy', gforces_data.get('gy2', 0.0)) if isinstance(gforces_data, dict) else 0.0
                    vert_g = gforces_data.get('gz', gforces_data.get('gz2', 0.0)) if isinstance(gforces_data, dict) else 0.0

                    damage_data = getattr(test_vehicle, 'damage', None) or {}
                    damage_total = damage_data.get('damage', 0.0) if isinstance(damage_data, dict) else 0.0

                    timer_data = getattr(test_vehicle, 'timer', None) or {}
                    sim_time = timer_data.get('time', time.time() - start_time) if isinstance(timer_data, dict) else time.time() - start_time

                    # Cone-hit heuristic - see TRACK_CONE_AI_INTEGRATION.md §5:
                    # not a real collision event, just "damage jumped, and
                    # here's the nearest cone at that moment."
                    if last_damage_total is not None and (damage_total - last_damage_total) > TRACK_CONE_HIT_DAMAGE_DELTA_THRESHOLD:
                        nearest_cone, nearest_dist = None, None
                        for cone in cone_positions:
                            d = math.hypot(cone["x"] - pos[0], cone["y"] - pos[1])
                            if nearest_dist is None or d < nearest_dist:
                                nearest_dist, nearest_cone = d, cone
                        if nearest_cone is not None and nearest_dist <= TRACK_CONE_HIT_DISTANCE_THRESHOLD_M:
                            cone_hits.append({
                                "elapsed_s": time.time() - start_time,
                                "cone": nearest_cone,
                                "distance_to_cone_m": nearest_dist,
                                "damage_delta": damage_total - last_damage_total,
                            })
                    last_damage_total = damage_total

                    telemetry["elapsed_s"].append(time.time() - start_time)
                    telemetry["sim_time_s"].append(sim_time)
                    telemetry["pos_x"].append(pos[0])
                    telemetry["pos_y"].append(pos[1])
                    telemetry["pos_z"].append(pos[2])
                    telemetry["speed_mps"].append(speed)
                    telemetry["distance_along_track_m"].append(distance_along)
                    telemetry["cross_track_error_m"].append(cross_track)
                    telemetry["lat_g"].append(lat_g)
                    telemetry["long_g"].append(long_g)
                    telemetry["vert_g"].append(vert_g)
                    telemetry["damage_total"].append(damage_total)

                    now = time.time()
                    if now - last_chart_update_time >= LIVE_CHART_UPDATE_INTERVAL_S:
                        last_chart_update_time = now
                        live_fig, live_ax = plt.subplots(figsize=(8, 3))
                        live_ax.plot(telemetry["elapsed_s"], telemetry["cross_track_error_m"], color="crimson", linewidth=1.2)
                        live_ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
                        live_ax.set_xlabel("Time (s)")
                        live_ax.set_ylabel("Cross-track error (m)")
                        live_ax.set_title(f"Live: Cross-Track Error - {distance_along:.0f}m / {total_length:.0f}m")
                        live_ax.grid(True, linestyle="--", alpha=0.6)
                        live_fig.tight_layout()
                        live_chart_placeholder.pyplot(live_fig)
                        plt.close(live_fig)

                    if finish_reached:
                        break
                    # Secondary/belt-and-suspenders check - see docstring.
                    if not track.closed_loop and distance_along >= total_length - 1.0:
                        finish_reached = True
                        break
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= DISCONNECT_FAILURE_LIMIT:
                    connection_lost = True
                    break

            time.sleep(BEAMNG_POLL_INTERVAL)

        live_chart_placeholder.empty()

        if connection_lost:
            return {
                "error": "Lost connection to BeamNG during the track run.",
                "connection_lost": True,
                "partial_telemetry": telemetry,
                "finish_checkpoints_passed": sum(checkpoints_passed),
                "finish_checkpoints_required": len(finish_checkpoints),
            }

        try:
            test_vehicle.ai.set_mode('disabled')
            test_vehicle.control(throttle=0.0, brake=1.0)
            beamng_client.pause()
        except Exception:
            pass

        # A shortfall here means execute_script fell behind schedule and
        # skipped nodes (see TRACK_CONE_AI_INTEGRATION.md §4/§9) - flag it
        # rather than silently presenting a truncated lap as a clean one.
        script_shortfall = max(0.0, total_length - distance_along) if not track.closed_loop else None

        return {
            "status": "Success",
            "telemetry": telemetry,
            "cone_hits": cone_hits,
            "total_track_length_m": total_length,
            "final_distance_along_track_m": distance_along,
            "script_shortfall_m": script_shortfall,
            "sample_count": len(telemetry["elapsed_s"]),
            "finish_line_crossed": finish_reached,
            "run_timeout_s": max_duration_s,
            "stopped_reason": "finish_line" if finish_reached else "timeout",
            "finish_checkpoints": finish_checkpoints,
            "finish_checkpoints_passed": sum(checkpoints_passed),
            "finish_checkpoints_required": len(finish_checkpoints),
            "rejected_finish_crossings": rejected_finish_crossings,
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ==========================================
# 3. STREAMLIT USER INTERFACE DASHBOARD
# ==========================================
st.set_page_config(page_title="AI Suspension Tuning Advisor", layout="wide")
st.title("🚗 AI-Powered Suspension Tuning Advisor & Validation Suite")
st.markdown("Optimize vehicle ride comfort using **Gaussian Process Bayesian Optimization** and validate findings against **BeamNG** telemetry.")

simulator = HalfCarPitchSimulator()

# Initialize State Variables
if "fs_rate" not in st.session_state: st.session_state.fs_rate = BO_INITIAL_GUESS[0]
if "f_damp" not in st.session_state:  st.session_state.f_damp = BO_INITIAL_GUESS[1]
if "rs_rate" not in st.session_state: st.session_state.rs_rate = BO_INITIAL_GUESS[2]
if "r_damp" not in st.session_state:  st.session_state.r_damp = BO_INITIAL_GUESS[3]

# Perform Auto-Detection on Load
if "bng_home_path" not in st.session_state:
    st.session_state.bng_home_path = detect_beamng_home()

# Holds a live BeamNGpy client/vehicle across Streamlit reruns so a finished
# test can stay open (paused) in-game until the user explicitly terminates
# it, instead of the game auto-closing the instant the test loop ends.
if "active_bng_client" not in st.session_state:
    st.session_state.active_bng_client = None
if "active_bng_vehicle" not in st.session_state:
    st.session_state.active_bng_vehicle = None
if "last_bng_result" not in st.session_state:
    st.session_state.last_bng_result = None

# Tracks the scenario "start menu" state: True once load_beamng_scenario()
# has successfully loaded the level and spawned the vehicle but the timed
# test hasn't been started yet. Flips back to False once run_beamng_test()
# runs (or a session is terminated), so the UI knows whether to show
# "Load Scenario", the in-game "waiting for accelerator" state, or the
# finished results.
if "bng_scenario_loaded" not in st.session_state:
    st.session_state.bng_scenario_loaded = False

# --- Track / Cone AI-driving session state (separate from the suspension-
# bump validation state above - see TRACK_CONE_AI_INTEGRATION.md) ---
if "track_bng_client" not in st.session_state:
    st.session_state.track_bng_client = None
if "track_bng_vehicle" not in st.session_state:
    st.session_state.track_bng_vehicle = None
if "parsed_track" not in st.session_state:
    st.session_state.parsed_track = None
if "track_parse_error" not in st.session_state:
    st.session_state.track_parse_error = None
if "track_scenario_result" not in st.session_state:
    st.session_state.track_scenario_result = None
if "last_track_run_result" not in st.session_state:
    st.session_state.last_track_run_result = None

st.sidebar.header("Optimization Profile")
selected_profile_name = st.sidebar.selectbox(
    "Select Target Objective",
    ["Comfort (Smooth Ride)", "Track Focused (Stiff & Responsive)", "Balanced", "Custom"],
    filter_mode=None
)

# Effective suspension search bounds used by the sliders below and by the
# Bayesian Optimization loop. Every preset uses the global defaults; only
# the Custom profile can override them (see below).
effective_front_spring_min, effective_front_spring_max = FRONT_SPRING_MIN, FRONT_SPRING_MAX
effective_front_damping_min, effective_front_damping_max = FRONT_DAMPING_MIN, FRONT_DAMPING_MAX
effective_rear_spring_min, effective_rear_spring_max = REAR_SPRING_MIN, REAR_SPRING_MAX
effective_rear_damping_min, effective_rear_damping_max = REAR_DAMPING_MIN, REAR_DAMPING_MAX

if selected_profile_name == "Custom":
    with st.sidebar.expander("Custom Objective Weights", expanded=True):
        st.caption(
            "Weights apply to normalized (dimensionless) response terms, so "
            "values are comparable across terms - start near the Balanced "
            "preset's defaults and nudge from there."
        )

        # --- Search range override ---
        st.markdown("**Suspension search range**")
        spring_abs_min, spring_abs_max = CUSTOM_SPRING_RANGE_ABSOLUTE_LIMITS
        damping_abs_min, damping_abs_max = CUSTOM_DAMPING_RANGE_ABSOLUTE_LIMITS
        st.caption(
            "Overrides the slider ranges below and the Bayesian Optimization "
            f"search space. Kept within {spring_abs_min:,}-{spring_abs_max:,} N/m "
            f"for springs and {damping_abs_min:,}-{damping_abs_max:,} Ns/m for "
            "damping so the physics model isn't pushed into degenerate "
            "territory (e.g. a zero spring rate)."
        )

        def _validated_range(min_val, max_val, default_min, default_max, axis_label):
            """Falls back to the global default range (with a warning) if the
            user's min/max are inverted or equal, rather than letting an
            empty/invalid range reach the sliders or the optimizer."""
            if min_val >= max_val:
                st.warning(f"{axis_label}: min must be less than max - using the default range instead.")
                return default_min, default_max
            return min_val, max_val

        range_col_front, range_col_rear = st.columns(2)
        with range_col_front:
            st.caption("Front")
            custom_front_spring_min = st.number_input(
                "Front Spring Min (N/m)", min_value=spring_abs_min, max_value=spring_abs_max,
                value=FRONT_SPRING_MIN, step=FRONT_SPRING_STEP, key="custom_front_spring_min",
            )
            custom_front_spring_max = st.number_input(
                "Front Spring Max (N/m)", min_value=spring_abs_min, max_value=spring_abs_max,
                value=FRONT_SPRING_MAX, step=FRONT_SPRING_STEP, key="custom_front_spring_max",
            )
            custom_front_damping_min = st.number_input(
                "Front Damping Min (Ns/m)", min_value=damping_abs_min, max_value=damping_abs_max,
                value=FRONT_DAMPING_MIN, step=FRONT_DAMPING_STEP, key="custom_front_damping_min",
            )
            custom_front_damping_max = st.number_input(
                "Front Damping Max (Ns/m)", min_value=damping_abs_min, max_value=damping_abs_max,
                value=FRONT_DAMPING_MAX, step=FRONT_DAMPING_STEP, key="custom_front_damping_max",
            )
        with range_col_rear:
            st.caption("Rear")
            custom_rear_spring_min = st.number_input(
                "Rear Spring Min (N/m)", min_value=spring_abs_min, max_value=spring_abs_max,
                value=REAR_SPRING_MIN, step=REAR_SPRING_STEP, key="custom_rear_spring_min",
            )
            custom_rear_spring_max = st.number_input(
                "Rear Spring Max (N/m)", min_value=spring_abs_min, max_value=spring_abs_max,
                value=REAR_SPRING_MAX, step=REAR_SPRING_STEP, key="custom_rear_spring_max",
            )
            custom_rear_damping_min = st.number_input(
                "Rear Damping Min (Ns/m)", min_value=damping_abs_min, max_value=damping_abs_max,
                value=REAR_DAMPING_MIN, step=REAR_DAMPING_STEP, key="custom_rear_damping_min",
            )
            custom_rear_damping_max = st.number_input(
                "Rear Damping Max (Ns/m)", min_value=damping_abs_min, max_value=damping_abs_max,
                value=REAR_DAMPING_MAX, step=REAR_DAMPING_STEP, key="custom_rear_damping_max",
            )

        effective_front_spring_min, effective_front_spring_max = _validated_range(
            custom_front_spring_min, custom_front_spring_max, FRONT_SPRING_MIN, FRONT_SPRING_MAX, "Front Spring Range"
        )
        effective_front_damping_min, effective_front_damping_max = _validated_range(
            custom_front_damping_min, custom_front_damping_max, FRONT_DAMPING_MIN, FRONT_DAMPING_MAX, "Front Damping Range"
        )
        effective_rear_spring_min, effective_rear_spring_max = _validated_range(
            custom_rear_spring_min, custom_rear_spring_max, REAR_SPRING_MIN, REAR_SPRING_MAX, "Rear Spring Range"
        )
        effective_rear_damping_min, effective_rear_damping_max = _validated_range(
            custom_rear_damping_min, custom_rear_damping_max, REAR_DAMPING_MIN, REAR_DAMPING_MAX, "Rear Damping Range"
        )

        # --- Objective/penalty weights, each toggleable ---
        custom_weights = {}

        st.markdown("**Ride objective**")
        st.caption(
            "Uncheck a category to exclude it from the cost score entirely "
            "for this run (its weight is treated as 0, independent of the "
            "number shown - so you can flip it back on without retyping)."
        )
        for field_key, field_label, field_min, field_max, field_step, field_help in CUSTOM_PROFILE_OBJECTIVE_FIELDS:
            default_weight = float(CUSTOM_PROFILE_DEFAULTS[field_key])
            enable_col, weight_col = st.columns([1, 3])
            with enable_col:
                term_enabled = st.checkbox(
                    "Use", value=default_weight > 0, key=f"custom_enable_{field_key}",
                    help=f"Include \u201c{field_label}\u201d in the cost score.",
                )
            with weight_col:
                weight_value = st.number_input(
                    field_label,
                    min_value=field_min,
                    max_value=field_max,
                    value=default_weight,
                    step=field_step,
                    help=field_help,
                    key=f"custom_{field_key}",
                    disabled=not term_enabled,
                )
            custom_weights[field_key] = weight_value if term_enabled else 0.0

        st.markdown("**Realism penalties**")
        for field_key, field_label, field_min, field_max, field_step, field_help in CUSTOM_PROFILE_REALISM_FIELDS:
            default_value = float(CUSTOM_PROFILE_DEFAULTS[field_key])
            if field_key.endswith("_penalty"):
                # Penalty *weights* get the same enable/disable treatment as
                # the objective terms above.
                enable_col, weight_col = st.columns([1, 3])
                with enable_col:
                    term_enabled = st.checkbox(
                        "Use", value=default_value > 0, key=f"custom_enable_{field_key}",
                        help=f"Include \u201c{field_label}\u201d in the cost score.",
                    )
                with weight_col:
                    weight_value = st.number_input(
                        field_label,
                        min_value=field_min,
                        max_value=field_max,
                        value=default_value,
                        step=field_step,
                        help=field_help,
                        key=f"custom_{field_key}",
                        disabled=not term_enabled,
                    )
                custom_weights[field_key] = weight_value if term_enabled else 0.0
            else:
                # Targets/tolerances/floors (damping_ratio_target, etc.) are
                # parameters, not independently-weighted categories, so they
                # don't get a checkbox - they only matter while the penalty
                # they belong to is enabled above.
                custom_weights[field_key] = st.number_input(
                    field_label,
                    min_value=field_min,
                    max_value=field_max,
                    value=default_value,
                    step=field_step,
                    help=field_help,
                    key=f"custom_{field_key}",
                )

    selected_profile = custom_weights
else:
    selected_profile = selected_profile_name

st.sidebar.header("Suspension Parameters")
# Clamp any already-set slider values into the (possibly just-changed)
# effective range before rendering the sliders, so switching profiles or
# tightening a Custom range never leaves a stored value outside its slider's
# min/max (Streamlit raises if it does).
if "fs_rate" in st.session_state:
    st.session_state.fs_rate = min(max(st.session_state.fs_rate, effective_front_spring_min), effective_front_spring_max)
if "f_damp" in st.session_state:
    st.session_state.f_damp = min(max(st.session_state.f_damp, effective_front_damping_min), effective_front_damping_max)
if "rs_rate" in st.session_state:
    st.session_state.rs_rate = min(max(st.session_state.rs_rate, effective_rear_spring_min), effective_rear_spring_max)
if "r_damp" in st.session_state:
    st.session_state.r_damp = min(max(st.session_state.r_damp, effective_rear_damping_min), effective_rear_damping_max)

front_spring_input = st.sidebar.slider("Front Spring Rate (N/m)", effective_front_spring_min, effective_front_spring_max, key="fs_rate", step=FRONT_SPRING_STEP)
front_damping_input = st.sidebar.slider("Front Damping (Ns/m)", effective_front_damping_min, effective_front_damping_max, key="f_damp", step=FRONT_DAMPING_STEP)
rear_spring_input = st.sidebar.slider("Rear Spring Rate (N/m)", effective_rear_spring_min, effective_rear_spring_max, key="rs_rate", step=REAR_SPRING_STEP)
rear_damping_input = st.sidebar.slider("Rear Damping (Ns/m)", effective_rear_damping_min, effective_rear_damping_max, key="r_damp", step=REAR_DAMPING_STEP)

st.sidebar.markdown("---")
st.sidebar.header("AI Optimization Control")
run_optimization_button = st.sidebar.button("Run Bayesian Optimization Loop")

column_left, column_right = st.columns(2)

with column_left:
    st.subheader("Python Physics Surrogate Model")
    
    current_cost, time_axis, bounce_data, front_height_data, rear_height_data = simulator.simulate_bump(
        front_spring_input, front_damping_input, rear_spring_input, rear_damping_input,
        profile=selected_profile
    )

    score_col, info_col = st.columns([6, 1])
    with score_col:
        st.metric(label=f"Calculated Cost Score ({selected_profile_name})", value=f"{current_cost:.4f}")
    with info_col:
        st.write("")  # nudge the button down to roughly align with the metric value
        weights_for_display = resolve_profile_weights(selected_profile)
        if hasattr(st, "popover"):
            with st.popover("ℹ️", use_container_width=True):
                render_score_explanation(weights_for_display, selected_profile_name)
        else:
            # Older Streamlit versions without st.popover: fall back to an
            # expander so the explanation is still reachable by clicking.
            with st.expander("ℹ️"):
                render_score_explanation(weights_for_display, selected_profile_name)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_axis, front_height_data, label="Front Wheel Height", color="crimson", linewidth=1.5, linestyle="-.")
    ax.plot(time_axis, rear_height_data, label="Rear Wheel Height", color="darkorange", linewidth=1.5, linestyle="--")
    ax.plot(time_axis, bounce_data, label="Chassis Heave", color="royalblue", linewidth=2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Height from Static Ride Height (m)")
    ax.set_title("Simulated Response Over 10 Sequential Washboard Bumps")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

with column_right:
    st.subheader("BeamNG Game Validation Engine")
    
    current_bng_home = st.session_state.bng_home_path
    is_valid_path = validate_beamng_directory(current_bng_home)

    if is_valid_path:
        st.success(f"BeamNG Detected: `{current_bng_home}`")

        scenario_ready = (
            st.session_state.active_bng_client is not None
            and st.session_state.bng_scenario_loaded
        )

        if st.session_state.active_bng_client is None:
            # --- Scenario start menu, step 1: load the level ---------------
            # This is the one and only website button in the flow - it has
            # to be, since it's what launches the external BeamNG process in
            # the first place. Everything after this happens in-game: once
            # the level and car are loaded, the script blocks and waits for
            # the player to press the accelerator inside BeamNG itself
            # (wait_for_native_start) before the timed run begins, so
            # there's no separate "Start Test" button on the site.
            st.caption(
                "Loads the level and spawns the car. Once it's loaded, "
                "start the timed test by pressing the accelerator inside "
                "BeamNG - not from a button here - so level load time "
                "doesn't get counted in the average speed."
            )
            if st.button("📥 Load Scenario"):
                with st.spinner("Launching BeamNG and loading scenario..."):
                    load_result = load_beamng_scenario(bng_home=current_bng_home)

                if "error" in load_result:
                    st.session_state.last_bng_result = load_result
                    st.session_state.bng_scenario_loaded = False
                else:
                    client, vehicle = load_result["client"], load_result["vehicle"]
                    st.session_state.active_bng_client = client
                    st.session_state.active_bng_vehicle = vehicle
                    st.session_state.bng_scenario_loaded = True
                    st.session_state.last_bng_result = None

                    # Blocks until the player presses the accelerator in
                    # BeamNG (or the connection drops) - see
                    # _run_native_start_and_test() / wait_for_native_start().
                    # If it comes back without starting (only possible on a
                    # disconnect, since no timeout is set by default), the
                    # scenario just stays loaded - the "Resume Waiting"
                    # control below covers picking it back up after e.g. a
                    # page refresh.
                    _run_native_start_and_test(client, vehicle)
                st.rerun()

        elif scenario_ready:
            # --- Scenario is loaded but the blocking wait above got
            # interrupted (e.g. the browser tab was refreshed mid-wait) -
            # this is a fallback to resume it, not the primary flow. In
            # normal use the "Load Scenario" click above already waits for
            # and runs the test in one go.
            st.info(
                "✅ Scenario loaded and vehicle spawned. Press the "
                "accelerator in BeamNG to start the timed test."
            )
            resume_col, cancel_col = st.columns([2, 1])
            with resume_col:
                if st.button("🎮 Resume Waiting for BeamNG Start"):
                    client = st.session_state.active_bng_client
                    vehicle = st.session_state.active_bng_vehicle
                    _run_native_start_and_test(client, vehicle)
                    st.rerun()
            with cancel_col:
                if st.button("✖ Cancel"):
                    try:
                        st.session_state.active_bng_client.close()
                    except Exception:
                        pass
                    st.session_state.active_bng_client = None
                    st.session_state.active_bng_vehicle = None
                    st.session_state.bng_scenario_loaded = False
                    st.rerun()

        # Placed AFTER the load button's block (not before) so that when a
        # test just finished above, this reads the freshly-updated session
        # state in the SAME script pass and shows the button immediately -
        # no rerun needed. Only shown once the timed test has actually run -
        # while the scenario is merely loaded-and-waiting, the "Cancel"
        # button above covers closing the session instead.
        if st.session_state.active_bng_client is not None and not st.session_state.bng_scenario_loaded:
            st.info("A BeamNG test session is open and paused in-game.")
            if st.button("⏹ Terminate BeamNG Session"):
                try:
                    st.session_state.active_bng_client.close()
                except Exception:
                    pass
                st.session_state.active_bng_client = None
                st.session_state.active_bng_vehicle = None
                st.rerun()

        if st.session_state.last_bng_result is not None:
            validation_result = st.session_state.last_bng_result
            if "error" in validation_result:
                st.error(f"BeamNG Execution Error: {validation_result['error']}")
                if "traceback" in validation_result:
                    with st.expander("Full traceback"):
                        st.code(validation_result["traceback"])
            else:
                session_still_open = st.session_state.active_bng_client is not None
                if session_still_open:
                    st.success(
                        "Automated test lap completed successfully! BeamNG is now paused - "
                        "use 'Terminate BeamNG Session' above when you're done inspecting it."
                    )
                else:
                    st.success("Automated test lap completed successfully. (BeamNG session has been terminated.)")

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric(
                    label="Peak Vertical G-Force",
                    value=f"{validation_result['peak_vertical_g']:.2f} G",
                )
                metric_col2.metric(
                    label="Average Speed",
                    value=f"{validation_result.get('avg_speed_mps', 0.0):.1f} m/s",
                )
                metric_col3.metric(
                    label="Distance Traveled",
                    value=f"{validation_result.get('distance_traveled_m', 0.0):.1f} m",
                )
                st.caption(
                    f"{validation_result.get('sample_count', 0)} telemetry samples over "
                    f"{validation_result.get('test_duration_s', 0)}s test duration."
                )

                telemetry = validation_result.get("telemetry")
                if telemetry and telemetry.get("elapsed_s"):
                    # Up/down motion, not the top-down path: chassis height
                    # (world Z, straight from the State sensor) on top, plus
                    # the derived vertical G-force underneath. BeamNG.drive's
                    # free (non-tech) API doesn't expose true per-wheel
                    # suspension travel - that requires BeamNG.tech - so
                    # chassis height is the most direct real "up/down" signal
                    # available; it's the same underlying data the vertical
                    # G-force below is derived from.
                    telem_fig, (height_ax, accel_ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

                    height_ax.plot(telemetry["elapsed_s"], telemetry["pos_z"], color="royalblue", linewidth=1.3)
                    height_ax.set_ylabel("Height (m)")
                    height_ax.set_title("Chassis Height Over Time (Up/Down)")
                    height_ax.grid(True, linestyle="--", alpha=0.6)

                    accel_ax.plot(telemetry["elapsed_s"], telemetry["vertical_g"], color="crimson", linewidth=1.2)
                    accel_ax.set_xlabel("Time (seconds)")
                    accel_ax.set_ylabel("Vertical G-Force")
                    accel_ax.set_title("BeamNG Measured Vertical G-Force Over Time")
                    accel_ax.grid(True, linestyle="--", alpha=0.6)

                    telem_fig.tight_layout()
                    st.pyplot(telem_fig)
                    plt.close(telem_fig)

                    wheel_speeds = [v for v in telemetry.get("avg_wheel_av", []) if v is not None]
                    if wheel_speeds:
                        wheel_fig, wheel_ax = plt.subplots(figsize=(8, 3))
                        wheel_ax.plot(telemetry["elapsed_s"][:len(wheel_speeds)], wheel_speeds, color="darkorange", linewidth=1.3)
                        wheel_ax.set_xlabel("Time (seconds)")
                        wheel_ax.set_ylabel("Avg. Wheel Angular Velocity (rad/s)")
                        wheel_ax.set_title("Wheel Speed Over Time (from Electrics sensor)")
                        wheel_ax.grid(True, linestyle="--", alpha=0.6)
                        st.pyplot(wheel_fig)
                        plt.close(wheel_fig)
                    else:
                        st.caption(
                            "No per-wheel data was reported by the Electrics sensor for this "
                            "vehicle/build - showing chassis-level height and G-force only."
                        )

                    with st.expander("Raw telemetry samples"):
                        st.dataframe(telemetry)
    else:
        st.error("⚠️ BeamNG Home Directory Not Found or Invalid!")
        st.warning("Automated search could not find a valid BeamNG installation. Validation is halted.")
        
        st.write("Please manually choose an option:")
        btn_col1, btn_col2 = st.columns([1, 1])
        
        with btn_col1:
            if st.button("Browse & Select Game Folder"):
                folder = select_folder_gui()
                if folder:
                    if validate_beamng_directory(folder):
                        st.session_state.bng_home_path = folder
                        st.rerun()
                    else:
                        st.error("Selected directory is missing `BeamNG.drive.x64.exe` or `Bin64` executable files!")

        with btn_col2:
            manual_input_path = st.text_input("Or enter path directly:", value=current_bng_home if current_bng_home else "")
            if st.button("Validate Path"):
                if validate_beamng_directory(manual_input_path):
                    st.session_state.bng_home_path = manual_input_path
                    st.rerun()
                else:
                    st.error("Specified path is not a valid BeamNG folder.")

st.markdown("---")
with st.expander("🏁 Track / Cone AI Driving (Beta)", expanded=False):
    st.caption(
        "Load a track from a CSV/JSON centerline file, spawn cones along its "
        "boundary, and drive it with BeamNG's own AI (execute_script) - "
        "instead of the fixed-bump suspension test above. See "
        "TRACK_CONE_AI_INTEGRATION.md for the full design writeup, including "
        "which telemetry is actually free vs. BeamNG.tech-gated. This is a "
        "separate BeamNG session from the suspension validation above; only "
        "run one at a time."
    )

    if BEAMNG_CONE_PROP_MODEL is None:
        st.warning(
            "`BEAMNG_CONE_PROP_MODEL` is not set near the top of this file - "
            "cone spawning will fail until you confirm the exact cone/pylon "
            "prop name for your installed BeamNG version in-game and set it "
            "(see TRACK_CONE_AI_INTEGRATION.md §0/§3). The track can still "
            "be parsed/previewed and driven without cones in the meantime."
        )

    track_upload_col, track_settings_col = st.columns([2, 1])
    with track_upload_col:
        track_file = st.file_uploader("Track file (CSV or JSON)", type=["csv", "json"])
    with track_settings_col:
        cone_spacing_input = st.number_input("Cone spacing (m)", min_value=0.5, value=DEFAULT_CONE_SPACING_M, step=0.5)

    # Track files are authored in a small local frame (see
    # TRACK_WORLD_ORIGIN_XY above) - these let the track be placed anywhere
    # on the map instead of always landing on BEAMNG_SPAWN_POS. Re-parses
    # below whenever these change, since the offset is baked into the
    # points at parse time.
    origin_col_x, origin_col_y = st.columns(2)
    with origin_col_x:
        track_origin_x = st.number_input("Track World Origin X", value=float(TRACK_WORLD_ORIGIN_XY[0]), format="%.3f")
    with origin_col_y:
        track_origin_y = st.number_input("Track World Origin Y", value=float(TRACK_WORLD_ORIGIN_XY[1]), format="%.3f")

    if track_file is not None:
        try:
            parsed = load_track_file(
                track_file, filename_hint=track_file.name,
                origin_xy=(track_origin_x, track_origin_y),
            )
            st.session_state.parsed_track = parsed
            st.session_state.track_parse_error = None
        except Exception as parse_exc:
            st.session_state.parsed_track = None
            st.session_state.track_parse_error = str(parse_exc)

    if st.session_state.track_parse_error:
        st.error(f"Could not parse track file: {st.session_state.track_parse_error}")

    parsed_track = st.session_state.parsed_track
    if parsed_track is not None:
        try:
            preview_cones = generate_all_cone_positions(
                parsed_track, cone_spacing_m=cone_spacing_input, run_up_m=RUN_UP_DISTANCE_M
            )
            preview_script = build_ai_script(parsed_track, run_up_m=RUN_UP_DISTANCE_M)
        except Exception as preview_exc:
            preview_cones, preview_script = [], []
            st.error(f"Could not build cone layout / AI script for this track: {preview_exc}")

        st.write(
            f"**{parsed_track.name}** - {len(parsed_track.points)} centerline points, "
            f"{'closed loop' if parsed_track.closed_loop else 'open track'}, "
            f"{len(preview_cones)} cones at {cone_spacing_input:.1f} m spacing, "
            f"{len(preview_script)}-node AI script."
        )

        preview_fig, preview_ax = plt.subplots(figsize=(7, 6))
        cx = [p.x for p in parsed_track.points]
        cy = [p.y for p in parsed_track.points]
        preview_ax.plot(cx, cy, color="royalblue", linewidth=1.5, label="Centerline")
        for side, color in (("left", "orange"), ("right", "orangered")):
            side_x = [c["x"] for c in preview_cones if c["side"] == side]
            side_y = [c["y"] for c in preview_cones if c["side"] == side]
            preview_ax.scatter(side_x, side_y, s=14, color=color, label=f"Cones ({side})")
        # Draw the start/finish gate(s) the AI run will auto-stop on (see
        # compute_finish_line() / FINISH_LINE_HALF_WIDTH_M) so it's visible
        # before running, not just discovered after the fact in the result.
        try:
            preview_finish_line = compute_finish_line(parsed_track)
            gate_specs = [("start", preview_finish_line["start_point"], preview_finish_line["start_tangent"], "green")]
            if not preview_finish_line["same_line"]:
                gate_specs.append(("finish", preview_finish_line["finish_point"], preview_finish_line["finish_tangent"], "black"))
            for label, gate_point, gate_tangent, gate_color in gate_specs:
                nx, ny = -gate_tangent[1], gate_tangent[0]
                gx = [gate_point[0] - nx * FINISH_LINE_HALF_WIDTH_M, gate_point[0] + nx * FINISH_LINE_HALF_WIDTH_M]
                gy = [gate_point[1] - ny * FINISH_LINE_HALF_WIDTH_M, gate_point[1] + ny * FINISH_LINE_HALF_WIDTH_M]
                preview_ax.plot(gx, gy, color=gate_color, linewidth=2.5, linestyle=":",
                                 label=f"{label.capitalize()} gate" + (" (start+finish)" if preview_finish_line["same_line"] else ""))
        except Exception:
            pass  # preview-only visualization; never block the rest of the preview on it
        preview_ax.set_aspect("equal", adjustable="datalim")
        preview_ax.set_xlabel("X (m)")
        preview_ax.set_ylabel("Y (m)")
        preview_ax.set_title("Track Preview")
        preview_ax.grid(True, linestyle="--", alpha=0.5)
        preview_ax.legend(loc="best", fontsize=8)
        st.pyplot(preview_fig)
        plt.close(preview_fig)

        current_bng_home = st.session_state.bng_home_path
        is_valid_path = validate_beamng_directory(current_bng_home)

        if not is_valid_path:
            st.error("⚠️ BeamNG Home Directory Not Found or Invalid! Fix this in the panel above before running a track.")
        else:
            run_col1, run_col2 = st.columns(2)
            with run_col1:
                if st.session_state.track_bng_client is None:
                    if st.button("📥 Load Track Scenario"):
                        with st.spinner("Launching BeamNG and loading track scenario..."):
                            scenario_result = load_beamng_track_scenario(
                                current_bng_home, parsed_track, cone_spacing_m=cone_spacing_input
                            )
                        st.session_state.track_scenario_result = scenario_result
                        st.rerun()
                else:
                    if st.button("🚦 Run AI Lap"):
                        client = st.session_state.track_bng_client
                        vehicle = st.session_state.track_bng_vehicle
                        cone_positions = (st.session_state.track_scenario_result or {}).get("cone_positions", [])
                        # This used to be an obfuscated resume() call
                        # wrapped in `except Exception: pass` with no
                        # camera handling at all - which is exactly why
                        # the track flow's camera never followed the car
                        # even though the top ("Load Scenario") flow's did.
                        camera_warning = resume_and_focus_camera(client, vehicle)
                        if camera_warning:
                            st.warning(
                                f"⚠️ Camera may still be in free-cam: {camera_warning} "
                                "You can switch views manually in-game (default key is "
                                "usually F7/Numpad keys depending on your BeamNG bindings)."
                            )
                        with st.spinner("Running AI-driven track lap..."):
                            run_result = run_beamng_track_test(client, vehicle, parsed_track, preview_script, cone_positions)
                        st.session_state.last_track_run_result = run_result
                        st.rerun()
            with run_col2:
                if st.session_state.track_bng_client is not None:
                    if st.button("⏹ Terminate Track Session"):
                        try:
                            st.session_state.track_bng_client.close()
                        except Exception:
                            pass
                        st.session_state.track_bng_client = None
                        st.session_state.track_bng_vehicle = None
                        st.session_state.track_scenario_result = None
                        st.rerun()

        scenario_result = st.session_state.track_scenario_result
        if scenario_result is not None:
            if "error" in scenario_result:
                st.error(f"Track scenario error: {scenario_result['error']}")
                if "traceback" in scenario_result:
                    with st.expander("Full traceback"):
                        st.code(scenario_result["traceback"])
            else:
                st.success(
                    f"Scenario loaded. Requested {scenario_result.get('cones_requested', 0)} cones; "
                    f"{scenario_result.get('cones_confirmed', 0)} spawned successfully "
                    "(this is an exact count of successful vehicles.spawn() calls for the "
                    "cones, not a guess)."
                )
                if scenario_result.get("cone_warning"):
                    st.warning(scenario_result["cone_warning"])
                if scenario_result.get("camera_warning"):
                    st.warning(
                        f"⚠️ Camera may still be in free-cam: {scenario_result['camera_warning']} "
                        "You can switch views manually in-game (default key is "
                        "usually F7/Numpad keys depending on your BeamNG bindings)."
                    )

        run_result = st.session_state.last_track_run_result
        if run_result is not None:
            if "error" in run_result:
                st.error(f"Track run error: {run_result['error']}")
                if "traceback" in run_result:
                    with st.expander("Full traceback"):
                        st.code(run_result["traceback"])
            else:
                telemetry = run_result["telemetry"]
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Sample count", run_result.get("sample_count", 0))
                m2.metric("Distance driven", f"{run_result.get('final_distance_along_track_m', 0.0):.1f} m")
                shortfall = run_result.get("script_shortfall_m")
                m3.metric("Script shortfall", f"{shortfall:.1f} m" if shortfall is not None else "N/A (closed loop)")
                m4.metric("Inferred cone hits", len(run_result.get("cone_hits", [])))
                m5.metric("Finish line", "✅ Crossed" if run_result.get("finish_line_crossed") else "⏱️ Timed out")
                m6.metric(
                    "Checkpoints",
                    f"{run_result.get('finish_checkpoints_passed', 0)}/{run_result.get('finish_checkpoints_required', 0)}",
                )
                if not run_result.get("finish_line_crossed"):
                    st.warning(
                        f"The run ended via the {run_result.get('run_timeout_s', 0.0):.0f}s fallback "
                        "timeout without crossing the finish line - the car likely crashed, got "
                        "stuck, or fell too far behind the AI script's schedule to finish. Check "
                        "the driven-path plot below for where it stopped making progress."
                    )
                if run_result.get("rejected_finish_crossings"):
                    n_rejected = len(run_result["rejected_finish_crossings"])
                    st.info(
                        f"The car crossed the start/finish line {n_rejected} time(s) before it "
                        "had passed all of the required checkpoints, so those crossings weren't "
                        "counted as the finish (see FINISH_CHECKPOINT_COUNT in MAIN.py) - this is "
                        "expected if the car wobbles over the line while the run-up settles it "
                        "onto the path at the start of a run."
                    )
                if shortfall and shortfall > 1.0:
                    st.warning(
                        "The AI fell behind the script's schedule and skipped ahead "
                        f"(~{shortfall:.1f} m short of the full track) - see "
                        "TRACK_CONE_AI_INTEGRATION.md §4/§9. Treat this run's "
                        "telemetry as a shortcut lap, not a full clean one."
                    )

                if telemetry.get("elapsed_s"):
                    run_fig, run_ax = plt.subplots(figsize=(7, 6))
                    run_ax.plot(cx, cy, color="lightsteelblue", linewidth=1.2, label="Centerline")
                    for side, color in (("left", "moccasin"), ("right", "lightsalmon")):
                        side_x = [c["x"] for c in preview_cones if c["side"] == side]
                        side_y = [c["y"] for c in preview_cones if c["side"] == side]
                        run_ax.scatter(side_x, side_y, s=10, color=color)
                    sc = run_ax.scatter(
                        telemetry["pos_x"], telemetry["pos_y"], c=telemetry["cross_track_error_m"],
                        cmap="coolwarm", s=6, label="Driven path (colored by cross-track error)",
                    )
                    for hit in run_result.get("cone_hits", []):
                        run_ax.scatter([hit["cone"]["x"]], [hit["cone"]["y"]], marker="x", s=80, color="black")
                    run_fig.colorbar(sc, ax=run_ax, label="Cross-track error (m)")
                    run_ax.set_aspect("equal", adjustable="datalim")
                    run_ax.set_title("Driven Path vs. Track")
                    run_ax.legend(loc="best", fontsize=8)
                    st.pyplot(run_fig)
                    plt.close(run_fig)

                    telem_fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
                    axes[0].plot(telemetry["elapsed_s"], telemetry["speed_mps"], color="royalblue")
                    axes[0].set_ylabel("Speed (m/s)")
                    axes[0].grid(True, linestyle="--", alpha=0.5)
                    axes[1].plot(telemetry["elapsed_s"], telemetry["cross_track_error_m"], color="crimson")
                    axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
                    axes[1].set_ylabel("Cross-track error (m)")
                    axes[1].grid(True, linestyle="--", alpha=0.5)
                    axes[2].plot(telemetry["elapsed_s"], telemetry["lat_g"], label="Lateral G", color="darkorange")
                    axes[2].plot(telemetry["elapsed_s"], telemetry["damage_total"], label="Damage total", color="black", linewidth=0.8)
                    axes[2].set_ylabel("G / Damage")
                    axes[2].set_xlabel("Time (s)")
                    axes[2].legend(loc="best", fontsize=8)
                    axes[2].grid(True, linestyle="--", alpha=0.5)
                    telem_fig.tight_layout()
                    st.pyplot(telem_fig)
                    plt.close(telem_fig)

                    telemetry_csv = io.StringIO()
                    writer = csv.writer(telemetry_csv)
                    writer.writerow(telemetry.keys())
                    writer.writerows(zip(*telemetry.values()))
                    st.download_button(
                        "⬇ Download telemetry CSV", data=telemetry_csv.getvalue(),
                        file_name="track_run_telemetry.csv", mime="text/csv",
                    )

                    with st.expander("Raw telemetry samples"):
                        st.dataframe(telemetry)
                    if run_result.get("cone_hits"):
                        with st.expander("Inferred cone contacts (heuristic, not exact - see TRACK_CONE_AI_INTEGRATION.md §5)"):
                            st.dataframe(run_result["cone_hits"])

if run_optimization_button:
    st.markdown("---")
    st.subheader(f"Running Bayesian Optimization Loop ({selected_profile_name})...")

    # Search space for this run: the Custom profile's overridden range if one
    # is active (see effective_* above), otherwise the global defaults.
    active_search_low = [
        effective_front_spring_min, effective_front_damping_min,
        effective_rear_spring_min, effective_rear_damping_min,
    ]
    active_search_high = [
        effective_front_spring_max, effective_front_damping_max,
        effective_rear_spring_max, effective_rear_damping_max,
    ]
    active_search_bounds = list(zip(active_search_low, active_search_high))

    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_left, _ = st.columns([3, 1])
    optimization_chart = chart_left.empty()

    optimization_history = {
        "iteration": [],
        "score": [],
        "best_score": [],
        "front_spring": [],
        "front_damping": [],
        "rear_spring": [],
        "rear_damping": [],
    }
    total_evaluations = BO_NUM_INITIAL_SAMPLES + BO_TOTAL_ITERATIONS

    def record_evaluation(params, score):
        optimization_history["iteration"].append(len(optimization_history["iteration"]) + 1)
        optimization_history["score"].append(score)
        optimization_history["best_score"].append(min(optimization_history["score"]))
        optimization_history["front_spring"].append(params[0])
        optimization_history["front_damping"].append(params[1])
        optimization_history["rear_spring"].append(params[2])
        optimization_history["rear_damping"].append(params[3])

        progress = len(optimization_history["iteration"]) / total_evaluations
        progress_bar.progress(min(progress, 1.0))
        status_text.text(
            f"Evaluation {len(optimization_history['iteration'])}/{total_evaluations} | "
            f"Latest Score: {score:.4f} | Best Score: {optimization_history['best_score'][-1]:.4f}"
        )

        fig = plot_optimization_history(optimization_history)
        optimization_chart.pyplot(fig)
        plt.close(fig)
    
    # --- Step 1: Random initial exploration ---
    training_inputs_list = []
    training_scores_list = []

    for sample in np.random.uniform(
        low=active_search_low,
        high=active_search_high,
        size=(BO_NUM_INITIAL_SAMPLES, 4),
    ):
        sample_score = simulator.simulate_bump(*sample, profile=selected_profile)[0]
        training_inputs_list.append(sample)
        training_scores_list.append(sample_score)
        record_evaluation(sample, sample_score)

    training_inputs = np.array(training_inputs_list)
    training_scores = np.array(training_scores_list)
    
    gaussian_process = GaussianProcessRegressor(
        kernel=Matern(nu=BO_MATERN_NU), 
        n_restarts_optimizer=BO_RESTARTS, 
        random_state=BO_RANDOM_STATE
    )
    
    best_idx = np.argmin(training_scores)
    best_score_record = training_scores[best_idx]
    best_parameters_found = training_inputs[best_idx]
    guided_iterations_without_improvement = 0
    stopped_early = False

    # --- Step 2: Guided (Bayesian) search ---
    for iteration_index in range(BO_TOTAL_ITERATIONS):
        gaussian_process.fit(training_inputs, training_scores)
        
        def acquisition_function(candidate_params):
            reshaped_candidate = np.array(candidate_params).reshape(1, -1)
            predicted_mean, predicted_std = gaussian_process.predict(reshaped_candidate, return_std=True)
            return (predicted_mean - BO_LCB_KAPPA * predicted_std)[0]

        best_acq_val = float('inf')
        next_candidate_params = None
        
        random_starts = np.random.uniform(
            low=active_search_low,
            high=active_search_high,
            size=(10, 4)
        )
        
        for start_point in random_starts:
            res = minimize(acquisition_function, start_point, bounds=active_search_bounds, method='L-BFGS-B')
            if res.fun < best_acq_val:
                best_acq_val = res.fun
                next_candidate_params = res.x
        
        candidate_score, *_ = simulator.simulate_bump(*next_candidate_params, profile=selected_profile)
        
        training_inputs = np.vstack((training_inputs, next_candidate_params))
        training_scores = np.append(training_scores, candidate_score)
        
        if best_score_record - candidate_score >= BO_EARLY_STOPPING_MIN_DELTA:
            best_score_record = candidate_score
            best_parameters_found = next_candidate_params
            guided_iterations_without_improvement = 0
        else:
            guided_iterations_without_improvement += 1
            if candidate_score < best_score_record:
                best_score_record = candidate_score
                best_parameters_found = next_candidate_params

        record_evaluation(next_candidate_params, candidate_score)

        if guided_iterations_without_improvement >= BO_EARLY_STOPPING_PATIENCE:
            stopped_early = True
            break
        
    if stopped_early:
        st.info(
            f"Optimization stopped early after {len(optimization_history['iteration'])} evaluations "
            f"({guided_iterations_without_improvement} guided iterations without ≥{BO_EARLY_STOPPING_MIN_DELTA:.3f} improvement)."
        )
    st.success("🎉 Optimization Complete! Optimal setup identified:")
    st.write(f"- **Front Spring Rate:** {best_parameters_found[0]:.1f} N/m")
    st.write(f"- **Front Damping:** {best_parameters_found[1]:.1f} Ns/m")
    st.write(f"- **Rear Spring Rate:** {best_parameters_found[2]:.1f} N/m")
    st.write(f"- **Rear Damping:** {best_parameters_found[3]:.1f} Ns/m")

    def apply_optimal_settings():
        st.session_state.fs_rate = int(round(best_parameters_found[0]))
        st.session_state.f_damp = int(round(best_parameters_found[1]))
        st.session_state.rs_rate = int(round(best_parameters_found[2]))
        st.session_state.r_damp = int(round(best_parameters_found[3]))

    st.button("Apply Optimal Parameters to Sliders", on_click=apply_optimal_settings)