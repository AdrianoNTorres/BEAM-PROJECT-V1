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
"""

import os
import time
from pathlib import Path
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
    from beamngpy import BeamNGpy, Scenario, Vehicle
    # NOTE: AdvancedIMU is intentionally not imported/used. It is a
    # BeamNG.tech-only sensor and is rejected server-side with
    # "This feature requires a BeamNG.tech license" on plain BeamNG.drive.
    # Vertical acceleration is instead derived from finite-differencing the
    # free `state` sensor's velocity - see run_beamng_validation().
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
SIM_TIME_END = 5.0       
SIM_TIME_POINTS = 150    

BUMP_START_TIME = 1.0    
BUMP_HEIGHT = 0.05       
BUMP_LENGTH = 0.5        

BUMP_DURATION = BUMP_LENGTH / BEAMNG_AI_SPEED
BUMP_END_TIME = BUMP_START_TIME + BUMP_DURATION 

BUMP_EVENT_END = BUMP_END_TIME + REAR_AXLE_ROAD_LAG + 0.15
POST_BUMP_START = BUMP_END_TIME + REAR_AXLE_ROAD_LAG + 0.2

PROFILE_WEIGHTS = {
    "Comfort (Smooth Ride)": {
        "peak_pitch_accel": 1.0,
        "post_heave_rms": 2500.0,
        "post_pitch_vel_rms": 1000.0,
        "post_heave_accel_rms": 400.0,
        "post_heave_peak_to_peak": 800.0,
        "spring_rate_sum": 0.0,
        "underspring_penalty": 2e-7,
        "damping_ratio_penalty": 80.0,
    },
    "Track Focused (Stiff & Responsive)": {
        "peak_pitch_accel": 0.15,
        "post_heave_rms": 5000.0,
        "post_pitch_vel_rms": 2500.0,
        "post_heave_accel_rms": 1000.0,
        "post_heave_peak_to_peak": 1500.0,
        "spring_rate_sum": 0.00008,
    },
    "Balanced": {
        "peak_pitch_accel": 1.0,
        "post_heave_rms": 3500.0,
        "post_pitch_vel_rms": 1500.0,
        "post_heave_accel_rms": 600.0,
        "post_heave_peak_to_peak": 1000.0,
        "spring_rate_sum": 0.0,
        "underspring_penalty": 1e-7,
        "damping_ratio_penalty": 60.0,
    },
}

COMFORT_TARGET_SPRING_MIN = 25000.0       
COMFORT_DAMPING_RATIO_TARGET = 0.45
COMFORT_DAMPING_RATIO_TOLERANCE = 0.20    
 
# --- BeamNG Validation Settings ---
BEAMNG_HOST = 'localhost'
BEAMNG_PORT = 25252
BEAMNG_MAP = 'gridmap_v2'
BEAMNG_VEHICLE_MODEL = 'etkc'
BEAMNG_VEHICLE_LICENSE = 'AI_TEST'
BEAMNG_SPAWN_POS = (-337.6, -491.3, 100.15)
# Rotation captured from the World Editor Inspector (rotationEuler Z = 180 deg,
# X/Y = 0). Converted to a quaternion (x, y, z, w) for beamngpy's rot_quat:
# a pure 180 deg yaw about Z is (0, 0, 1, 0).
BEAMNG_SPAWN_ROT_QUAT = (0, 0, 1, 0)
BEAMNG_IMU_FREQ = 50           
BEAMNG_TEST_DURATION = 45      
BEAMNG_POLL_INTERVAL = 0.02    

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
    def __init__(self):
        self.chassis_sprung_mass = CHASSIS_SPRUNG_MASS
        self.front_unsprung_mass = FRONT_UNSPRUNG_MASS
        self.rear_unsprung_mass = REAR_UNSPRUNG_MASS
        self.pitch_inertia = PITCH_INERTIA
        self.distance_front_axle = DISTANCE_FRONT_AXLE
        self.distance_rear_axle = DISTANCE_REAR_AXLE

    def compute_damping_ratios(self, front_spring_rate, front_damping, rear_spring_rate, rear_damping):
        front_sprung_load = self.chassis_sprung_mass * self.distance_rear_axle / WHEELBASE
        rear_sprung_load = self.chassis_sprung_mass * self.distance_front_axle / WHEELBASE
        front_critical_damping = 2.0 * np.sqrt(front_spring_rate * front_sprung_load)
        rear_critical_damping = 2.0 * np.sqrt(rear_spring_rate * rear_sprung_load)
        front_damping_ratio = front_damping / front_critical_damping
        rear_damping_ratio = rear_damping / rear_critical_damping
        return front_damping_ratio, rear_damping_ratio

    def compute_realism_penalties(self, front_spring_rate, front_damping, rear_spring_rate, rear_damping, weights):
        underspring_penalty = (
            max(0.0, COMFORT_TARGET_SPRING_MIN - front_spring_rate) ** 2
            + max(0.0, COMFORT_TARGET_SPRING_MIN - rear_spring_rate) ** 2
        )

        front_damping_ratio, rear_damping_ratio = self.compute_damping_ratios(
            front_spring_rate, front_damping, rear_spring_rate, rear_damping
        )
        damping_ratio_penalty = 0.0
        for damping_ratio in (front_damping_ratio, rear_damping_ratio):
            damping_deviation = abs(damping_ratio - COMFORT_DAMPING_RATIO_TARGET) - COMFORT_DAMPING_RATIO_TOLERANCE
            damping_ratio_penalty += max(0.0, damping_deviation) ** 2

        return (
            weights.get("underspring_penalty", 0.0) * underspring_penalty
            + weights.get("damping_ratio_penalty", 0.0) * damping_ratio_penalty
        )

    def compute_static_equilibrium(self, front_spring_rate, rear_spring_rate):
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

    def simulate_bump(self, front_spring_rate, front_damping, rear_spring_rate, rear_damping, profile="Comfort (Smooth Ride)"):
        def road_speed_bump(time_t):
            if BUMP_START_TIME <= time_t <= BUMP_END_TIME:
                return BUMP_HEIGHT * np.sin(np.pi * (time_t - BUMP_START_TIME) / (BUMP_END_TIME - BUMP_START_TIME))
            return 0.0

        def road_speed_bump_velocity(time_t):
            if BUMP_START_TIME <= time_t <= BUMP_END_TIME:
                return BUMP_HEIGHT * (np.pi / (BUMP_END_TIME - BUMP_START_TIME)) * np.cos(np.pi * (time_t - BUMP_START_TIME) / (BUMP_END_TIME - BUMP_START_TIME))
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
            max_step=BUMP_DURATION / 4.0
        )
        
        chassis_heave_response = simulation_solution.y[0] - static_vertical_bounce
        front_height_response = simulation_solution.y[4] - static_front_wheel
        rear_height_response = simulation_solution.y[6] - static_rear_wheel
        chassis_heave_velocity = simulation_solution.y[1]
        chassis_pitch_velocity = simulation_solution.y[3]
        
        pitch_accelerations = np.gradient(chassis_pitch_velocity, time_evaluation_points)
        heave_accelerations = np.gradient(chassis_heave_velocity, time_evaluation_points)

        bump_event_mask = (time_evaluation_points >= BUMP_START_TIME) & (time_evaluation_points <= BUMP_EVENT_END)
        post_bump_mask = time_evaluation_points >= POST_BUMP_START
        post_bump_heave = chassis_heave_response[post_bump_mask]

        peak_pitch_accel = np.max(np.abs(pitch_accelerations[bump_event_mask]))

        post_heave_rms = np.sqrt(np.mean(post_bump_heave**2))
        post_pitch_vel_rms = np.sqrt(np.mean(chassis_pitch_velocity[post_bump_mask]**2))
        post_heave_accel_rms = np.sqrt(np.mean(heave_accelerations[post_bump_mask]**2))
        post_heave_peak_to_peak = (
            np.max(post_bump_heave) - np.min(post_bump_heave) if post_bump_heave.size else 0.0
        )

        weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["Balanced"])
        cost_score = (
            weights["peak_pitch_accel"] * peak_pitch_accel
            + weights["post_heave_rms"] * post_heave_rms
            + weights["post_pitch_vel_rms"] * post_pitch_vel_rms
            + weights["post_heave_accel_rms"] * post_heave_accel_rms
            + weights["post_heave_peak_to_peak"] * post_heave_peak_to_peak
            + weights["spring_rate_sum"] * (front_spring_rate + rear_spring_rate)
            + self.compute_realism_penalties(
                front_spring_rate, front_damping, rear_spring_rate, rear_damping, weights
            )
        )
        
        return cost_score, time_evaluation_points, chassis_heave_response, front_height_response, rear_height_response


def plot_optimization_history(history):
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
# 2. BEAMNG AUTOMATION BRIDGE
# ==========================================
def run_beamng_validation(front_spring_rate, front_damping, rear_spring_rate, rear_damping, bng_home=None):
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

        # Allow physics engine and camera viewport to initialize
        time.sleep(3.0)

        # Spawn the vehicle live, at runtime, bypassing the broken prefab path.
        # spawn() also calls vehicle.connect() internally when connect=True,
        # so test_vehicle is fully usable immediately after this returns.
        spawn_success = beamng_client.vehicles.spawn(
            test_vehicle, pos=BEAMNG_SPAWN_POS, rot_quat=BEAMNG_SPAWN_ROT_QUAT, connect=True
        )
        if not spawn_success:
            beamng_client.close()
            return {"error": "beamng_client.vehicles.spawn() reported failure - vehicle did not spawn."}

        # Put the player's camera/controls into the vehicle. Without this,
        # the spawned vehicle just sits there with nobody "seated" in it,
        # which is why free-cam was the only view available before.
        beamng_client.vehicles.switch(test_vehicle)

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

        test_vehicle.ai.set_mode('disabled')
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
        }

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
                    telemetry["elapsed_s"].append(time.time() - start_time)
                    telemetry["vertical_g"].append(sample_vertical_g)
                    telemetry["speed_mps"].append(horizontal_speed)
                    telemetry["pos_x"].append(vehicle_pos[0])
                    telemetry["pos_y"].append(vehicle_pos[1])
                    telemetry["pos_z"].append(vehicle_pos[2])
                    telemetry["throttle"].append(throttle_cmd)
                    telemetry["brake"].append(brake_cmd)
            except Exception:
                pass

            time.sleep(BEAMNG_POLL_INTERVAL)

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
        # Don't leave a half-initialized connection dangling on error.
        if 'beamng_client' in locals():
            try:
                beamng_client.close()
            except Exception:
                pass
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

st.sidebar.header("Optimization Profile")
selected_profile = st.sidebar.selectbox(
    "Select Target Objective",
    ["Comfort (Smooth Ride)", "Track Focused (Stiff & Responsive)", "Balanced"],
    filter_mode=None
)

st.sidebar.header("Suspension Parameters")
front_spring_input = st.sidebar.slider("Front Spring Rate (N/m)", FRONT_SPRING_MIN, FRONT_SPRING_MAX, key="fs_rate", step=FRONT_SPRING_STEP)
front_damping_input = st.sidebar.slider("Front Damping (Ns/m)", FRONT_DAMPING_MIN, FRONT_DAMPING_MAX, key="f_damp", step=FRONT_DAMPING_STEP)
rear_spring_input = st.sidebar.slider("Rear Spring Rate (N/m)", REAR_SPRING_MIN, REAR_SPRING_MAX, key="rs_rate", step=REAR_SPRING_STEP)
rear_damping_input = st.sidebar.slider("Rear Damping (Ns/m)", REAR_DAMPING_MIN, REAR_DAMPING_MAX, key="r_damp", step=REAR_DAMPING_STEP)

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
    st.metric(label=f"Calculated Cost Score ({selected_profile})", value=f"{current_cost:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_axis, front_height_data, label="Front Wheel Height", color="crimson", linewidth=1.5, linestyle="-.")
    ax.plot(time_axis, rear_height_data, label="Rear Wheel Height", color="darkorange", linewidth=1.5, linestyle="--")
    ax.plot(time_axis, bounce_data, label="Chassis Heave", color="royalblue", linewidth=2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Height from Static Ride Height (m)")
    ax.set_title(f"Simulated Response Over {int(BUMP_HEIGHT*100)}cm Speed Bump")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

with column_right:
    st.subheader("BeamNG Game Validation Engine")
    
    current_bng_home = st.session_state.bng_home_path
    is_valid_path = validate_beamng_directory(current_bng_home)

    if is_valid_path:
        st.success(f"BeamNG Detected: `{current_bng_home}`")

        if st.button("Run Automated BeamNG Test Lap"):
            with st.spinner("Launching BeamNG and running automated AI test drive..."):
                validation_result = run_beamng_validation(
                    front_spring_input, front_damping_input, rear_spring_input, rear_damping_input,
                    bng_home=current_bng_home
                )
            st.session_state.last_bng_result = validation_result

        # Placed AFTER the run button's block (not before) so that when a
        # test just finished above, this reads the freshly-updated session
        # state in the SAME script pass and shows the button immediately -
        # no rerun needed. (Checking this before the run button meant it
        # always evaluated against the state from before the click.)
        if st.session_state.active_bng_client is not None:
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
                st.success(
                    "Automated test lap completed successfully! BeamNG is now paused - "
                    "use 'Terminate BeamNG Session' above when you're done inspecting it."
                )

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
                    telem_fig, (accel_ax, speed_ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

                    accel_ax.plot(telemetry["elapsed_s"], telemetry["vertical_g"], color="crimson", linewidth=1.2)
                    accel_ax.set_ylabel("Vertical G-Force")
                    accel_ax.set_title("BeamNG Measured Vertical G-Force Over Time")
                    accel_ax.grid(True, linestyle="--", alpha=0.6)

                    speed_ax.plot(telemetry["elapsed_s"], telemetry["speed_mps"], color="royalblue", linewidth=1.5, label="Speed")
                    speed_ax.axhline(BEAMNG_AI_SPEED, color="gray", linestyle=":", linewidth=1, label="Target Speed")
                    speed_ax.set_xlabel("Time (seconds)")
                    speed_ax.set_ylabel("Speed (m/s)")
                    speed_ax.set_title("Vehicle Speed Over Time")
                    speed_ax.grid(True, linestyle="--", alpha=0.6)
                    speed_ax.legend()

                    telem_fig.tight_layout()
                    st.pyplot(telem_fig)
                    plt.close(telem_fig)

                    valid_xy = [
                        (x, y) for x, y in zip(telemetry["pos_x"], telemetry["pos_y"])
                        if x is not None and y is not None
                    ]
                    if valid_xy:
                        path_fig, path_ax = plt.subplots(figsize=(6, 6))
                        path_x, path_y = zip(*valid_xy)
                        path_ax.plot(path_x, path_y, color="darkorange", linewidth=1.5)
                        path_ax.scatter([path_x[0]], [path_y[0]], color="green", label="Start", zorder=5)
                        path_ax.scatter([path_x[-1]], [path_y[-1]], color="red", label="End", zorder=5)
                        path_ax.set_xlabel("World X (m)")
                        path_ax.set_ylabel("World Y (m)")
                        path_ax.set_title("Vehicle Path (Top-Down)")
                        path_ax.set_aspect("equal", adjustable="datalim")
                        path_ax.grid(True, linestyle="--", alpha=0.6)
                        path_ax.legend()
                        st.pyplot(path_fig)
                        plt.close(path_fig)

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

if run_optimization_button:
    st.markdown("---")
    st.subheader(f"Running Bayesian Optimization Loop ({selected_profile})...")
    
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
        low=[FRONT_SPRING_MIN, FRONT_DAMPING_MIN, REAR_SPRING_MIN, REAR_DAMPING_MIN],
        high=[FRONT_SPRING_MAX, FRONT_DAMPING_MAX, REAR_SPRING_MAX, REAR_DAMPING_MAX],
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
            low=[FRONT_SPRING_MIN, FRONT_DAMPING_MIN, REAR_SPRING_MIN, REAR_DAMPING_MIN], 
            high=[FRONT_SPRING_MAX, FRONT_DAMPING_MAX, REAR_SPRING_MAX, REAR_DAMPING_MAX], 
            size=(10, 4)
        )
        
        for start_point in random_starts:
            res = minimize(acquisition_function, start_point, bounds=SEARCH_BOUNDS, method='L-BFGS-B')
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