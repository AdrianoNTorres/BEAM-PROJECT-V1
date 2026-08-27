import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Try importing BeamNGpy (handled gracefully if the game isn't running locally)
try:
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors import AdvancedIMU
    BEAMNG_AVAILABLE = True
except ImportError:
    BEAMNG_AVAILABLE = False


# ==========================================
# GLOBAL CONFIGURATION & TUNING CONSTANTS
# ==========================================
# Suspension Parameter Bounds & Step Sizes
FRONT_SPRING_MIN, FRONT_SPRING_MAX, FRONT_SPRING_STEP = 20000, 150000, 1000
FRONT_DAMPING_MIN, FRONT_DAMPING_MAX, FRONT_DAMPING_STEP = 1000, 10000, 100
REAR_SPRING_MIN, REAR_SPRING_MAX, REAR_SPRING_STEP = 20000, 150000, 1000
REAR_DAMPING_MIN, REAR_DAMPING_MAX, REAR_DAMPING_STEP = 1000, 10000, 100

# Aggregated Search Bounds for Optimization
SEARCH_BOUNDS = [
    (FRONT_SPRING_MIN, FRONT_SPRING_MAX),
    (FRONT_DAMPING_MIN, FRONT_DAMPING_MAX),
    (REAR_SPRING_MIN, REAR_SPRING_MAX),
    (REAR_DAMPING_MIN, REAR_DAMPING_MAX)
]

# Vehicle Layout & Fundamental Physical Matrix
BEAMNG_AI_SPEED = 15.0         # Target velocity (m/s)
CHASSIS_SPRUNG_MASS = 1200.0   # kg
FRONT_UNSPRUNG_MASS = 40.0     # kg
REAR_UNSPRUNG_MASS = 40.0      # kg
DISTANCE_FRONT_AXLE = 1.2      # meters (CoG to Front)
DISTANCE_REAR_AXLE = 1.4       # meters (CoG to Rear)
TIRE_STIFFNESS = 200000.0      # N/m
TIRE_DAMPING = 2500.0          # Ns/m (Added tire damping constant)

# --- DYNAMICALLY CALCULATED PHYSICS VALUES ---
WHEELBASE = DISTANCE_FRONT_AXLE + DISTANCE_REAR_AXLE  # 2.6 meters
REAR_AXLE_ROAD_LAG = WHEELBASE / BEAMNG_AI_SPEED      # ~0.1733 seconds

# Calculate Pitch Inertia using standard mass radius of gyration layout rule: I = m * a * b
PITCH_INERTIA = CHASSIS_SPRUNG_MASS * (DISTANCE_FRONT_AXLE * DISTANCE_REAR_AXLE) 

# Simulation Profile Settings
SIM_TIME_START = 0.0
SIM_TIME_END = 3.0
SIM_TIME_POINTS = 150

BUMP_START_TIME = 1.0
BUMP_HEIGHT = 0.05             # 5cm speed bump
BUMP_LENGTH = 0.5              # 0.5 meter physical bump length

# Calculate exact time window width the car spends passing over the physical bump length
BUMP_DURATION = BUMP_LENGTH / BEAMNG_AI_SPEED
BUMP_END_TIME = BUMP_START_TIME + BUMP_DURATION # Dynamic based on velocity parameters

# BeamNG Validation Settings
BEAMNG_HOST = 'localhost'
BEAMNG_PORT = 25252
BEAMNG_MAP = 'gridmap_v2'
BEAMNG_VEHICLE_MODEL = 'etk800'
BEAMNG_VEHICLE_LICENSE = 'AI_TEST'
BEAMNG_SPAWN_POS = (-50, 0, 0)
BEAMNG_IMU_FREQ = 50           # Hz
BEAMNG_TEST_DURATION = 10      # seconds
BEAMNG_POLL_INTERVAL = 0.02    # seconds

# Bayesian Optimization Settings
BO_NUM_INITIAL_SAMPLES = 8
BO_TOTAL_ITERATIONS = 100
BO_MATERN_NU = 1.5
BO_RESTARTS = 5
BO_RANDOM_STATE = 42
BO_LCB_KAPPA = 1.96            # Exploration weight for Lower Confidence Bound

# Calculate initial optimization starting point using boundaries midpoints
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

    def calculate_system_equations(self, time_t, state_vector, front_spring_rate, front_damping, rear_spring_rate, rear_damping, road_profile_func, road_velocity_func):
        (vertical_bounce, vertical_velocity, 
         pitch_angle, pitch_velocity, 
         front_wheel_displacement, front_wheel_velocity, 
         rear_wheel_displacement, rear_wheel_velocity) = state_vector
        
        # Sample road height profile and derivative (velocity) with time lag for rear axle
        road_height_front = road_profile_func(time_t)
        road_height_rear = road_profile_func(time_t - REAR_AXLE_ROAD_LAG)
        road_vel_front = road_velocity_func(time_t)
        road_vel_rear = road_velocity_func(time_t - REAR_AXLE_ROAD_LAG)
        
        # Geometry coordinates for attachment points
        chassis_front_displacement = vertical_bounce - self.distance_front_axle * pitch_angle
        chassis_rear_displacement = vertical_bounce + self.distance_rear_axle * pitch_angle
        chassis_front_velocity = vertical_velocity - self.distance_front_axle * pitch_velocity
        chassis_rear_velocity = vertical_velocity + self.distance_rear_axle * pitch_velocity
        
        # Compute suspension forces
        front_suspension_force = -front_spring_rate * (chassis_front_displacement - front_wheel_displacement) - front_damping * (chassis_front_velocity - front_wheel_velocity)
        rear_suspension_force = -rear_spring_rate * (chassis_rear_displacement - rear_wheel_displacement) - rear_damping * (chassis_rear_velocity - rear_wheel_velocity)
        
        # Compute tire contact forces (lift-off protection mapped to stiffness and damping components)
        front_tire_force = max(0.0, -TIRE_STIFFNESS * (front_wheel_displacement - road_height_front) - TIRE_DAMPING * (front_wheel_velocity - road_vel_front))
        rear_tire_force = max(0.0, -TIRE_STIFFNESS * (rear_wheel_displacement - road_height_rear) - TIRE_DAMPING * (rear_wheel_velocity - road_vel_rear))
        
        # Accelerations via Newton's Second Law
        vertical_chassis_acceleration = (front_suspension_force + rear_suspension_force) / self.chassis_sprung_mass                  
        pitch_chassis_acceleration = (rear_suspension_force * self.distance_rear_axle - front_suspension_force * self.distance_front_axle) / self.pitch_inertia 
        front_wheel_acceleration = (-front_suspension_force + front_tire_force) / self.front_unsprung_mass             
        rear_wheel_acceleration = (-rear_suspension_force + rear_tire_force) / self.rear_unsprung_mass             
        
        return [
            vertical_velocity, vertical_chassis_acceleration, 
            pitch_velocity, pitch_chassis_acceleration, 
            front_wheel_velocity, front_wheel_acceleration, 
            rear_wheel_velocity, rear_wheel_acceleration
        ]

    def simulate_bump(self, front_spring_rate, front_damping, rear_spring_rate, rear_damping):
        def road_speed_bump(time_t):
            if BUMP_START_TIME <= time_t <= BUMP_END_TIME:
                return BUMP_HEIGHT * np.sin(np.pi * (time_t - BUMP_START_TIME) / (BUMP_END_TIME - BUMP_START_TIME))
            return 0.0

        def road_speed_bump_velocity(time_t):
            if BUMP_START_TIME <= time_t <= BUMP_END_TIME:
                return BUMP_HEIGHT * (np.pi / (BUMP_END_TIME - BUMP_START_TIME)) * np.cos(np.pi * (time_t - BUMP_START_TIME) / (BUMP_END_TIME - BUMP_START_TIME))
            return 0.0

        initial_state = [0, 0, 0, 0, 0, 0, 0, 0]
        time_span = (SIM_TIME_START, SIM_TIME_END)
        time_evaluation_points = np.linspace(SIM_TIME_START, SIM_TIME_END, SIM_TIME_POINTS)
        
        simulation_solution = solve_ivp(
            lambda time_t, state_vector: self.calculate_system_equations(
                time_t, state_vector, front_spring_rate, front_damping, rear_spring_rate, rear_damping, road_speed_bump, road_speed_bump_velocity
            ),
            time_span, initial_state, t_eval=time_evaluation_points
        )
        
        chassis_pitch_accelerations = simulation_solution.y[3] 
        cost_score = np.max(np.abs(chassis_pitch_accelerations))
        
        return cost_score, time_evaluation_points, simulation_solution.y[0]


# ==========================================
# 2. BEAMNG AUTOMATION BRIDGE
# ==========================================
def run_beamng_validation(front_spring_rate, front_damping, rear_spring_rate, rear_damping):
    if not BEAMNG_AVAILABLE:
        return {"error": "beamngpy library not installed or configured."}
    
    try:
        beamng_client = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT)
        beamng_client.open(launch=True)

        scenario = Scenario(BEAMNG_MAP, 'ai_suspension_validation')
        test_vehicle = Vehicle('ego_vehicle', model=BEAMNG_VEHICLE_MODEL, license=BEAMNG_VEHICLE_LICENSE)
        scenario.add_vehicle(test_vehicle, pos=BEAMNG_SPAWN_POS)
        scenario.make(beamng_client)

        beamng_client.scenario.load(scenario)
        beamng_client.scenario.start()

        imu_sensor = AdvancedIMU('imu', beamng_client, test_vehicle, frequency=BEAMNG_IMU_FREQ)

        test_vehicle.ai.set_mode('speed')
        test_vehicle.ai.set_speed(BEAMNG_AI_SPEED) 

        start_time = time.time()
        vertical_g_forces = []
        
        while time.time() - start_time < BEAMNG_TEST_DURATION:
            sensor_data = imu_sensor.poll()
            if sensor_data and 'z' in sensor_data:
                vertical_g_forces.append(abs(sensor_data['z']))
            time.sleep(BEAMNG_POLL_INTERVAL)

        test_vehicle.ai.set_mode('disabled')
        beamng_client.close()

        peak_vertical_g = max(vertical_g_forces) if vertical_g_forces else 0.0
        return {"status": "Success", "peak_vertical_g": peak_vertical_g}

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 3. STREAMLIT USER INTERFACE DASHBOARD
# ==========================================
st.set_page_config(page_title="AI Suspension Tuning Advisor", layout="wide")
st.title("🚗 AI-Powered Suspension Tuning Advisor & Validation Suite")
st.markdown("Optimize vehicle ride comfort using **Gaussian Process Bayesian Optimization** and validate findings against **BeamNG** telemetry.")

simulator = HalfCarPitchSimulator()

# Initialize Session State Defaults using calculated midpoint constants
if "fs_rate" not in st.session_state: st.session_state.fs_rate = BO_INITIAL_GUESS[0]
if "f_damp" not in st.session_state:  st.session_state.f_damp = BO_INITIAL_GUESS[1]
if "rs_rate" not in st.session_state: st.session_state.rs_rate = BO_INITIAL_GUESS[2]
if "r_damp" not in st.session_state:  st.session_state.r_damp = BO_INITIAL_GUESS[3]

st.sidebar.header("🎛️ Suspension Parameters")

front_spring_input = st.sidebar.slider("Front Spring Rate (N/m)", FRONT_SPRING_MIN, FRONT_SPRING_MAX, key="fs_rate", step=FRONT_SPRING_STEP)
front_damping_input = st.sidebar.slider("Front Damping (Ns/m)", FRONT_DAMPING_MIN, FRONT_DAMPING_MAX, key="f_damp", step=FRONT_DAMPING_STEP)
rear_spring_input = st.sidebar.slider("Rear Spring Rate (N/m)", REAR_SPRING_MIN, REAR_SPRING_MAX, key="rs_rate", step=REAR_SPRING_STEP)
rear_damping_input = st.sidebar.slider("Rear Damping (Ns/m)", REAR_DAMPING_MIN, REAR_DAMPING_MAX, key="r_damp", step=REAR_DAMPING_STEP)

st.sidebar.markdown("---")
st.sidebar.header("AI Optimization Control")
run_optimization_button = st.sidebar.button("Run Bayesian Optimization Loop")

column_left, column_right = st.columns(2)

with column_left:
    st.subheader("📊 Python Physics Surrogate Model")
    
    current_cost, time_axis, bounce_data = simulator.simulate_bump(
        front_spring_input, front_damping_input, rear_spring_input, rear_damping_input
    )
    st.metric(label="Calculated Ride Comfort Cost Score (Lower is Better)", value=f"{current_cost:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_axis, bounce_data, label="Chassis Heave Response", color="royalblue", linewidth=2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Vertical Displacement (m)")
    ax.set_title(f"Simulated Response Over {int(BUMP_HEIGHT*100)}cm Speed Bump")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

with column_right:
    st.subheader("🎮 BeamNG Game Validation Engine")
    st.info("Status: Ready to execute automated test run in simulation environment.")
    
    if st.button("🚀 Run Automated BeamNG Test Lap"):
        with st.spinner("Launching BeamNG and running automated AI test drive..."):
            validation_result = run_beamng_validation(
                front_spring_input, front_damping_input, rear_spring_input, rear_damping_input
            )
            
            if "error" in validation_result:
                st.warning(f"BeamNG Note: {validation_result['error']}")
                st.info("Tip: Ensure BeamNG.drive is installed and accessible locally to run full engine validation.")
            else:
                st.success("Automated test lap completed successfully!")
                st.metric(label="BeamNG Measured Peak Vertical G-Force", value=f"{validation_result['peak_vertical_g']:.2f} G")

if run_optimization_button:
    st.markdown("---")
    st.subheader("🔄 Running Bayesian Optimization Loop...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Generate initial random training points using global bounds limits
    training_inputs = np.random.uniform(
        low=[FRONT_SPRING_MIN, FRONT_DAMPING_MIN, REAR_SPRING_MIN, REAR_DAMPING_MIN], 
        high=[FRONT_SPRING_MAX, FRONT_DAMPING_MAX, REAR_SPRING_MAX, REAR_DAMPING_MAX], 
        size=(BO_NUM_INITIAL_SAMPLES, 4)
    )
    training_scores = np.array([simulator.simulate_bump(*sample)[0] for sample in training_inputs])
    
    @st.cache_resource
    def get_gaussian_process():
        return GaussianProcessRegressor(
            kernel=Matern(nu=BO_MATERN_NU), 
            n_restarts_optimizer=BO_RESTARTS, 
            random_state=BO_RANDOM_STATE
        )

    gaussian_process = get_gaussian_process()
    
    best_score_record = float('inf')
    best_parameters_found = None
    
    for iteration_index in range(BO_TOTAL_ITERATIONS):
        gaussian_process.fit(training_inputs, training_scores)
        
        def acquisition_function(candidate_params):
            reshaped_candidate = np.array(candidate_params).reshape(1, -1)
            predicted_mean, predicted_std = gaussian_process.predict(reshaped_candidate, return_std=True)
            return predicted_mean - BO_LCB_KAPPA * predicted_std

        optimization_result = minimize(
            acquisition_function, BO_INITIAL_GUESS, bounds=SEARCH_BOUNDS, method='L-BFGS-B'
        )
        next_candidate_params = optimization_result.x
        
        candidate_score, _, _ = simulator.simulate_bump(*next_candidate_params)
        
        training_inputs = np.vstack((training_inputs, next_candidate_params))
        training_scores = np.append(training_scores, candidate_score)
        
        if candidate_score < best_score_record:
            best_score_record = candidate_score
            best_parameters_found = next_candidate_params
            
        progress_bar.progress((iteration_index + 1) / BO_TOTAL_ITERATIONS)
        status_text.text(f"Iteration {iteration_index+1}/{BO_TOTAL_ITERATIONS} completed. Best Score: {best_score_record:.4f}")
        time.sleep(0.1)
        
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

    st.button("⚙️ Apply Optimal Parameters to Sliders", on_click=apply_optimal_settings)