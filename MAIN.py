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

import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from scipy.integrate import solve_ivp   # numerical ODE integrator (Runge-Kutta family)
from scipy.optimize import minimize, root  # `root` solves the nonlinear static-equilibrium equations;
                                            # `minimize` finds the best next point to sample during
                                            # Bayesian Optimization (minimizing the acquisition function).
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

# --- Suspension Parameter Bounds & Step Sizes ---
# These define the 4-dimensional search space that both the manual sliders
# and the Bayesian Optimizer operate over. Units:
#   Spring rate  k  [N/m]   -- force per unit compression (Hooke's Law: F = k*x)
#   Damping rate c  [Ns/m]  -- force per unit compression *velocity* (F = c*v)
FRONT_SPRING_MIN, FRONT_SPRING_MAX, FRONT_SPRING_STEP = 20000, 150000, 1000
FRONT_DAMPING_MIN, FRONT_DAMPING_MAX, FRONT_DAMPING_STEP = 1000, 10000, 100
REAR_SPRING_MIN, REAR_SPRING_MAX, REAR_SPRING_STEP = 20000, 150000, 1000
REAR_DAMPING_MIN, REAR_DAMPING_MAX, REAR_DAMPING_STEP = 1000, 10000, 100

# Aggregated Search Bounds for Optimization.
# Order matches the parameter tuple used everywhere else in the file:
# (front_spring_rate, front_damping, rear_spring_rate, rear_damping).
# scipy.optimize.minimize expects bounds as a list of (low, high) pairs,
# one per dimension, in the same order as the parameter vector.
SEARCH_BOUNDS = [
    (FRONT_SPRING_MIN, FRONT_SPRING_MAX),
    (FRONT_DAMPING_MIN, FRONT_DAMPING_MAX),
    (REAR_SPRING_MIN, REAR_SPRING_MAX),
    (REAR_DAMPING_MIN, REAR_DAMPING_MAX)
]

# --- Vehicle Layout & Fundamental Physical Constants ---
BEAMNG_AI_SPEED = 15.0         # Target velocity (m/s) the car drives at, both in the
                                # math model's bump-timing and in the BeamNG AI driver.
CHASSIS_SPRUNG_MASS = 1200.0   # kg -- mass of the chassis body supported BY the springs
FRONT_UNSPRUNG_MASS = 40.0     # kg -- mass of the front wheel/tire/hub assembly, which
                                # is NOT cushioned by the springs (it sits below them)
REAR_UNSPRUNG_MASS = 40.0      # kg -- same, for the rear wheel assembly
DISTANCE_FRONT_AXLE = 1.2      # meters, horizontal distance from the Center of Gravity
                                # (CoG) to the front axle. Together with the rear
                                # distance this sets the weight distribution.
DISTANCE_REAR_AXLE = 1.4       # meters, CoG to rear axle.
TIRE_STIFFNESS = 200000.0      # N/m -- tire modeled as a simple linear spring
TIRE_DAMPING = 2500.0          # Ns/m -- small viscous damping added to the tire model
                                # (real tires have some internal energy dissipation)
GRAVITY = 9.81                 # m/s^2 -- standard gravitational acceleration

# --- DYNAMICALLY CALCULATED PHYSICS VALUES ---

# Wheelbase = total front-to-rear axle distance.
#   WHEELBASE = DISTANCE_FRONT_AXLE + DISTANCE_REAR_AXLE 
WHEELBASE = DISTANCE_FRONT_AXLE + DISTANCE_REAR_AXLE

# The rear wheels hit any given piece of road WHEELBASE meters, and therefore
# (WHEELBASE / speed) seconds, after the front wheels do (distance = speed * time,
# so time = distance / speed). This lag is what makes the pitch model realistic:
# the front axle "feels" the bump first, then the rear axle feels the *same*
# road profile shifted later in time.
#   REAR_AXLE_ROAD_LAG = WHEELBASE / BEAMNG_AI_SPEED 
REAR_AXLE_ROAD_LAG = WHEELBASE / BEAMNG_AI_SPEED 

# Pitch moment of inertia I (how hard it is to rotate the chassis about its
# CoG in the pitch axis). This uses the common half-car approximation
#   I = m * a * b
# where m is the sprung mass and a, b are the CoG-to-front and CoG-to-rear
# distances. This is equivalent to assuming the pitch radius of gyration
# k_p satisfies k_p^2 = a*b, a standard simplifying assumption in vehicle
# dynamics texts when detailed mass-distribution data isn't available
# (it reproduces realistic pitch/bounce mode coupling for typical cars).
#   PITCH_INERTIA = 1200.0 * (1.2 * 1.4) = 2016.0 kg*m^2
PITCH_INERTIA = CHASSIS_SPRUNG_MASS * (DISTANCE_FRONT_AXLE * DISTANCE_REAR_AXLE) 

# --- Simulation Profile Settings ---
SIM_TIME_START = 0.0     # seconds, start of the simulated time window
SIM_TIME_END = 3.0       # seconds, end of the simulated time window
SIM_TIME_POINTS = 150    # number of evenly spaced points at which the ODE
                          # solution is sampled/reported (affects plot resolution)

BUMP_START_TIME = 1.0    # seconds, when the front axle first reaches the bump
BUMP_HEIGHT = 0.05       # meters, 5 cm speed bump peak height
BUMP_LENGTH = 0.5        # meters, physical length (footprint) of the bump

# How long (in seconds) it takes the car, traveling at BEAMNG_AI_SPEED, to
# fully cross the bump's physical length:
#   time = distance / speed  =>  BUMP_DURATION = BUMP_LENGTH / BEAMNG_AI_SPEED
BUMP_DURATION = BUMP_LENGTH / BEAMNG_AI_SPEED
BUMP_END_TIME = BUMP_START_TIME + BUMP_DURATION # Dynamic based on velocity parameters

# --- BeamNG Validation Settings ---
BEAMNG_HOST = 'localhost'
BEAMNG_PORT = 25252
BEAMNG_MAP = 'gridmap_v2'
BEAMNG_VEHICLE_MODEL = 'etk800'
BEAMNG_VEHICLE_LICENSE = 'AI_TEST'
BEAMNG_SPAWN_POS = (-50, 0, 0)
BEAMNG_IMU_FREQ = 50           # Hz, IMU (Inertial Measurement Unit) sampling rate
BEAMNG_TEST_DURATION = 10      # seconds, total wall-clock time to poll the IMU for
BEAMNG_POLL_INTERVAL = 0.02    # seconds between IMU polls (50 Hz to match BEAMNG_IMU_FREQ)

# --- Bayesian Optimization (BO) Settings ---
# BO treats simulate_bump(...) as an expensive black-box function and tries
# to find its minimum using as few evaluations as possible, by fitting a
# probabilistic surrogate model (a Gaussian Process) to points seen so far.
BO_NUM_INITIAL_SAMPLES = 8     # number of random samples used to "seed" the GP
                                # before any guided search begins
BO_TOTAL_ITERATIONS = 100      # number of guided (GP-directed) samples to take
                                # after the initial random seeding
BO_MATERN_NU = 1.5             # smoothness parameter (nu) of the Matern kernel;
                                # nu=1.5 assumes the cost function is once
                                # differentiable but not perfectly smooth -- a
                                # common, robust default for physical systems
BO_RESTARTS = 5                # number of random restarts used when fitting the
                                # GP's kernel hyperparameters, to avoid local optima
BO_RANDOM_STATE = 42           # seed for reproducibility of the GP's internal
                                # random restarts
BO_LCB_KAPPA = 1.96            # Exploration weight (kappa) for the Lower Confidence
                                # Bound acquisition function:
                                #   LCB(x) = mu(x) - kappa * sigma(x)
                                # where mu(x) is the GP's predicted mean cost at x
                                # and sigma(x) is its predicted uncertainty (std dev).
                                # Larger kappa biases the search toward exploring
                                # uncertain regions; smaller kappa biases toward
                                # exploiting the current best-known region.
                                # kappa=1.96 corresponds to a ~95% confidence bound
                                # under a Gaussian assumption.

# Calculate initial optimization starting point using boundaries midpoints.
# This is just a fixed, deterministic starting point handed to L-BFGS-B when
# optimizing the acquisition function each BO iteration -- it is NOT the
# optimizer's final answer, only where its local search begins.
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
    """
    A classic 4-degree-of-freedom "half-car" (pitch-plane) suspension model.

    The system has 4 independent masses/motions (degrees of freedom), each
    described by a position and a velocity, for 8 total state variables:
        1. vertical_bounce        -- chassis heave (up/down) position [m]
        2. pitch_angle            -- chassis pitch (nose up/down rotation) [rad]
        3. front_wheel_displacement -- front unsprung mass vertical position [m]
        4. rear_wheel_displacement  -- rear unsprung mass vertical position [m]
    (each of the above also has a paired velocity in the ODE state vector).

    Sign convention: "up" is positive for all vertical displacements/forces.
    Pitch angle follows the right-hand rule about the lateral axis, such
    that positive pitch corresponds to the rear compressing more than the
    front (nose-up attitude), given how the geometry terms are written
    below (chassis_rear_displacement uses +pitch_angle, chassis_front uses
    -pitch_angle).

    Physical building blocks used throughout this class:
      * Hooke's Law (spring force):      F_spring = k * x
      * Viscous damping force:           F_damper = c * v
      * Newton's 2nd Law:                a = F_net / m
      * Rotational Newton's 2nd Law:     angular_acceleration = Torque / I
      * Torque of a force at distance d: Torque = F * d
    """

    def __init__(self):
        self.chassis_sprung_mass = CHASSIS_SPRUNG_MASS
        self.front_unsprung_mass = FRONT_UNSPRUNG_MASS
        self.rear_unsprung_mass = REAR_UNSPRUNG_MASS
        self.pitch_inertia = PITCH_INERTIA
        self.distance_front_axle = DISTANCE_FRONT_AXLE
        self.distance_rear_axle = DISTANCE_REAR_AXLE

    def compute_static_equilibrium(self, front_spring_rate, rear_spring_rate):
        """
        Solve for the flat-road, at-rest ride height and pitch angle under
        gravity alone (no bump, no motion) -- i.e. how far the springs
        naturally sag under the vehicle's own weight before any dynamic
        simulation starts.

        Mathematically this finds the state vector
            [vertical_bounce, pitch_angle, front_wheel_disp, rear_wheel_disp]
        for which every acceleration and the net pitch torque are exactly
        zero (static equilibrium: sum of forces = 0, sum of torques = 0).
        Because the tire force has a max(0, ...) unilateral clamp (a tire
        can only push, never pull), the resulting system is nonlinear, so
        it's solved numerically with `scipy.optimize.root` rather than
        with linear algebra.
        """
        def equilibrium_equations(state):
            vertical_bounce, pitch_angle, front_wheel_displacement, rear_wheel_displacement = state
            
            # Account for pitch in the static chassis attachment points.
            # Small-angle geometry: the vertical position of the chassis
            # mounting point at a horizontal distance d from the CoG is
            # approximately (bounce -/+ d * pitch_angle), i.e. a point
            # ahead of the CoG drops when pitch_angle rotates the nose down.
            chassis_front_displacement = vertical_bounce - self.distance_front_axle * pitch_angle
            chassis_rear_displacement = vertical_bounce + self.distance_rear_axle * pitch_angle
            
            # Spring compression = (wheel position) - (chassis attachment position).
            # Positive compression means the spring is squeezed shorter than
            # its free length (wheel has moved up relative to the chassis).
            front_compression = front_wheel_displacement - chassis_front_displacement
            rear_compression = rear_wheel_displacement - chassis_rear_displacement
            
            # Hooke's Law: spring force is proportional to compression.
            front_suspension_force = front_spring_rate * front_compression
            rear_suspension_force = rear_spring_rate * rear_compression
            
            # Tire modeled as a spring anchored to the (flat, height-0) road.
            # max(0, ...) makes it unilateral: a tire can only push the wheel
            # up when compressed against the road, never pull it down (a real
            # tire isn't bolted to the ground).
            front_tire_force = max(0.0, TIRE_STIFFNESS * (0.0 - front_wheel_displacement))
            rear_tire_force = max(0.0, TIRE_STIFFNESS * (0.0 - rear_wheel_displacement))
            
            # Net pitch torque about the CoG: rear suspension force acting at
            # +distance_rear_axle contributes +torque, front suspension force
            # acting at -distance_front_axle contributes -torque (Torque = F*d).
            # Net torque must be 0 at equilibrium (no angular acceleration).
            pitch_torque = (rear_suspension_force * self.distance_rear_axle - front_suspension_force * self.distance_front_axle)
            
            # Each returned value is "residual acceleration/torque" that
            # `root` drives to zero:
            #   - chassis vertical:  (spring forces)/mass - gravity == 0
            #   - chassis pitch:     net torque == 0
            #   - front wheel:       (tire - spring forces)/mass - gravity == 0
            #   - rear wheel:        (tire - spring forces)/mass - gravity == 0
            return [
                (front_suspension_force + rear_suspension_force) / self.chassis_sprung_mass - GRAVITY,
                pitch_torque, 
                (front_tire_force - front_suspension_force) / self.front_unsprung_mass - GRAVITY,
                (rear_tire_force - rear_suspension_force) / self.rear_unsprung_mass - GRAVITY,
            ]

        # Initial guess: level chassis (0 bounce, 0 pitch), wheels sagged
        # slightly below road height (-0.01 m) since gravity compresses the
        # tires against the road. `root` refines this guess numerically
        # (using a variant of Newton's method) until all four equations
        # above are ~zero.
        solution = root(equilibrium_equations, [0.0, 0.0, -0.01, -0.01])
        return solution.x # Now returns 4 variables

    def calculate_system_equations(self, time_t, state_vector, front_spring_rate, front_damping, rear_spring_rate, rear_damping, road_profile_func, road_velocity_func):
        """
        Right-hand side of the ODE system: given the current state and time,
        return the derivative of the state (velocities and accelerations).

        This is the function `scipy.integrate.solve_ivp` calls repeatedly to
        numerically integrate the system forward in time (i.e. it implements
        dy/dt = f(t, y) for the 8-element state vector y).

        The physics is the *dynamic* counterpart of `compute_static_equilibrium`:
        the same spring/damper/tire force structure, but now driven by a
        moving road profile (the bump) and including damping forces, with
        the resulting net force/torque producing real accelerations instead
        of being solved to zero.
        """
        (vertical_bounce, vertical_velocity, 
         pitch_angle, pitch_velocity, 
         front_wheel_displacement, front_wheel_velocity, 
         rear_wheel_displacement, rear_wheel_velocity) = state_vector
        
        # Sample road height (and its rate of change) under each axle.
        # The rear axle reaches any given point on the road REAR_AXLE_ROAD_LAG
        # seconds after the front axle (see WHEELBASE / speed calculation
        # above), so it experiences the identical profile function, just
        # evaluated at an earlier time argument (time_t - lag).
        road_height_front = road_profile_func(time_t)
        road_height_rear = road_profile_func(time_t - REAR_AXLE_ROAD_LAG)
        road_vel_front = road_velocity_func(time_t)
        road_vel_rear = road_velocity_func(time_t - REAR_AXLE_ROAD_LAG)
        
        # Geometry coordinates for attachment points (Upward = Positive).
        # Same small-angle approximation as in compute_static_equilibrium:
        # position/velocity of a chassis mount point at horizontal distance d
        # from the CoG = (bounce -/+ d * pitch) and its time-derivative
        # (velocity -/+ d * pitch_velocity), since d/dt(d * theta) = d * theta_dot
        # for a fixed distance d.
        chassis_front_displacement = vertical_bounce - self.distance_front_axle * pitch_angle
        chassis_rear_displacement = vertical_bounce + self.distance_rear_axle * pitch_angle
        chassis_front_velocity = vertical_velocity - self.distance_front_axle * pitch_velocity
        chassis_rear_velocity = vertical_velocity + self.distance_rear_axle * pitch_velocity
        
        # Compression distance (Positive when compressed: wheel higher than chassis attachment)
        front_compression = front_wheel_displacement - chassis_front_displacement
        rear_compression = rear_wheel_displacement - chassis_rear_displacement
        
        # Rate of change of compression = relative velocity between the
        # wheel and the chassis attachment point -- this is what the
        # damper force reacts to (F_damper = c * relative velocity).
        front_comp_velocity = front_wheel_velocity - chassis_front_velocity
        rear_comp_velocity = rear_wheel_velocity - chassis_rear_velocity
        
        # Internal Suspension Restoring Forces (Positive = pushing chassis up / wheel down).
        # This combines Hooke's Law (spring) and viscous damping (damper) in
        # parallel, the standard spring-damper (Kelvin-Voigt) model:
        #   F = k * x + c * dx/dt
        front_suspension_force = front_spring_rate * front_compression + front_damping * front_comp_velocity
        rear_suspension_force = rear_spring_rate * rear_compression + rear_damping * rear_comp_velocity
        
        # Tire contact forces (unilateral: can push wheel up, no pull when airborne).
        # Tire compression here is (road height - wheel height): if the road
        # rises faster than the wheel (going up a bump), compression grows
        # and the tire pushes back harder, exactly like the spring/damper
        # suspension force above but referenced to the road instead of the
        # chassis. max(0, ...) enforces that once the tire compression would
        # go negative (wheel above the road -- airborne / tire droop beyond
        # its unloaded length), the tire cannot exert a pulling force.
        front_tire_compression = road_height_front - front_wheel_displacement
        rear_tire_compression = road_height_rear - rear_wheel_displacement
        
        front_tire_comp_velocity = road_vel_front - front_wheel_velocity
        rear_tire_comp_velocity = road_vel_rear - rear_wheel_velocity
        
        front_tire_force = max(0.0, TIRE_STIFFNESS * front_tire_compression + TIRE_DAMPING * front_tire_comp_velocity)
        rear_tire_force = max(0.0, TIRE_STIFFNESS * rear_tire_compression + TIRE_DAMPING * rear_tire_comp_velocity)
        
        # Newton's 2nd Law (a = F_net / m) with gravity on sprung and
        # unsprung masses. The chassis is pushed up by both suspension
        # forces and pulled down by gravity.
        vertical_chassis_acceleration = (front_suspension_force + rear_suspension_force) / self.chassis_sprung_mass - GRAVITY
        
        # Pitch Torque: Rotational form of Newton's 2nd Law, angular
        # acceleration = net torque / moment of inertia. Rear suspension
        # force acting at +distance_rear_axle contributes positive
        # (nose-up) torque; front suspension force acting at
        # -distance_front_axle contributes negative (nose-down) torque,
        # matching the sign convention set up in the geometry section above.
        pitch_chassis_acceleration = (rear_suspension_force * self.distance_rear_axle - front_suspension_force * self.distance_front_axle) / self.pitch_inertia 
        
        # Wheels: tire force pushes the wheel mass upward, the reaction to
        # the suspension force pulls it downward (Newton's 3rd law -- the
        # suspension force that pushes the chassis up pulls the wheel down
        # by an equal and opposite amount), and gravity pulls it down.
        front_wheel_acceleration = (front_tire_force - front_suspension_force) / self.front_unsprung_mass - GRAVITY
        rear_wheel_acceleration = (rear_tire_force - rear_suspension_force) / self.rear_unsprung_mass - GRAVITY
        
        # solve_ivp expects dy/dt for every state variable, in the same
        # order as the state vector was unpacked above: each position's
        # derivative is simply the paired velocity, and each velocity's
        # derivative is the acceleration computed from F=ma above.
        return [
            vertical_velocity, vertical_chassis_acceleration, 
            pitch_velocity, pitch_chassis_acceleration, 
            front_wheel_velocity, front_wheel_acceleration, 
            rear_wheel_velocity, rear_wheel_acceleration
        ]

    def simulate_bump(self, front_spring_rate, front_damping, rear_spring_rate, rear_damping):
        """
        Run one full simulation of the car driving over a speed bump with
        the given spring/damper settings, and return a scalar "ride harshness"
        cost score along with the time series data needed to plot the response.

        Returns
        -------
        cost_score : float
            Peak absolute pitch acceleration [rad/s^2] observed during the
            bump event -- used as a proxy for perceived ride harshness (a
            sudden nose-up/nose-down jolt is what passengers actually feel
            and find uncomfortable, more so than raw bounce alone). Lower is
            better (a smoother, less jolting ride).
        time_evaluation_points : np.ndarray
            The time axis (seconds) at which the response was sampled.
        chassis_heave_response : np.ndarray
            Chassis vertical position relative to its static (at-rest) ride
            height [m], over time -- how much the body bounces up and down.
        """
        def road_speed_bump(time_t):
            """
            Road height profile h(t): a single half-sine-wave "bump" pulse.
            Zero everywhere except the short window [BUMP_START_TIME,
            BUMP_END_TIME], during which it traces out one arch of a sine
            wave from 0 up to BUMP_HEIGHT and back down to 0:
                h(t) = BUMP_HEIGHT * sin(pi * (t - t0) / duration)
            A half-sine gives a smooth (C0-continuous, zero-slope-at-the-
            edges... actually zero value but nonzero slope at the edges)
            bump shape, which is a common, simple approximation of a real
            speed bump's cross-section.
            """
            if BUMP_START_TIME <= time_t <= BUMP_END_TIME:
                return BUMP_HEIGHT * np.sin(np.pi * (time_t - BUMP_START_TIME) / (BUMP_END_TIME - BUMP_START_TIME))
            return 0.0

        def road_speed_bump_velocity(time_t):
            """
            Analytic derivative dh/dt of road_speed_bump(t), used to give
            the ODE the exact road *velocity* input (rather than having the
            solver numerically differentiate a possibly-noisy road profile).
            By the chain rule, differentiating
                h(t) = H * sin(pi * (t - t0) / T)
            with respect to t gives
                dh/dt = H * (pi / T) * cos(pi * (t - t0) / T)
            where H = BUMP_HEIGHT and T = BUMP_END_TIME - BUMP_START_TIME.
            """
            if BUMP_START_TIME <= time_t <= BUMP_END_TIME:
                return BUMP_HEIGHT * (np.pi / (BUMP_END_TIME - BUMP_START_TIME)) * np.cos(np.pi * (time_t - BUMP_START_TIME) / (BUMP_END_TIME - BUMP_START_TIME))
            return 0.0

        # Before simulating the bump, find the car's natural at-rest sag
        # under gravity on a flat road -- this becomes the starting point
        # (initial condition) for the dynamic simulation, and the reference
        # ("zero") level that heave is measured relative to afterward.
        static_vertical_bounce, static_pitch_angle, static_front_wheel, static_rear_wheel = self.compute_static_equilibrium(
            front_spring_rate, rear_spring_rate
        )
        
        # Initial state vector: at rest (all velocities = 0) in the static
        # equilibrium position computed above. Order matches the unpacking
        # in calculate_system_equations:
        # [bounce, bounce_vel, pitch, pitch_vel, front_wheel, front_wheel_vel,
        #  rear_wheel, rear_wheel_vel]
        initial_state = [
            static_vertical_bounce, 0, 
            static_pitch_angle, 0,
            static_front_wheel, 0, 
            static_rear_wheel, 0
        ]
        time_span = (SIM_TIME_START, SIM_TIME_END)
        time_evaluation_points = np.linspace(SIM_TIME_START, SIM_TIME_END, SIM_TIME_POINTS)
        
        # Numerically integrate the ODE system forward in time. solve_ivp
        # defaults to an adaptive-step Runge-Kutta method (RK45), but we cap
        # the maximum internal step size to a quarter of the bump's duration
        # (max_step=BUMP_DURATION/4.0) to make sure the solver doesn't step
        # clean over the short, sharp bump event and miss it entirely.
        simulation_solution = solve_ivp(
            lambda time_t, state_vector: self.calculate_system_equations(
                time_t, state_vector, front_spring_rate, front_damping, rear_spring_rate, rear_damping, road_speed_bump, road_speed_bump_velocity
            ),
            time_span, initial_state, t_eval=time_evaluation_points,
            max_step=BUMP_DURATION / 4.0
        )
        
        # y[0] is the vertical_bounce state trajectory; subtracting the
        # static equilibrium value re-references it to "heave from rest"
        # rather than absolute position, which is what's useful to plot.
        chassis_heave_response = simulation_solution.y[0] - static_vertical_bounce

        # y[3] is the pitch_velocity state trajectory. np.gradient performs
        # numerical differentiation (finite differences) to estimate the
        # pitch *acceleration* from the pitch *velocity* time series, since
        # acceleration wasn't stored directly by the ODE solver.
        chassis_pitch_velocity = simulation_solution.y[3]
        pitch_accelerations = np.gradient(chassis_pitch_velocity, time_evaluation_points)

        # The optimization objective: the single worst (largest magnitude)
        # pitch acceleration spike anywhere in the simulated time window.
        # This is what the Bayesian Optimizer below tries to minimize.
        cost_score = np.max(np.abs(pitch_accelerations))
        
        return cost_score, time_evaluation_points, chassis_heave_response


# ==========================================
# 2. BEAMNG AUTOMATION BRIDGE
# ==========================================
def run_beamng_validation(front_spring_rate, front_damping, rear_spring_rate, rear_damping):
    """
    Optional real-physics sanity check: launch BeamNG.drive (a full 3D
    vehicle physics game/simulator), spawn a car, drive it over a bump at
    BEAMNG_AI_SPEED using the game's built-in AI driver, and record the
    peak vertical G-force measured by an in-game IMU (accelerometer)
    sensor during the run.

    Note: the current suspension spring/damper parameters chosen in the UI
    are NOT actually applied to the BeamNG vehicle in this function (BeamNG
    vehicles have their own internal suspension config); this instead
    reports how the *stock* in-game vehicle responds, as a rough real-world
    reference point to compare the Python model's qualitative behavior
    against, rather than a 1:1 quantitative validation of the exact tuned
    parameters.

    Returns a dict with either {"status", "peak_vertical_g"} on success or
    {"error": <message>} if BeamNG isn't available or something goes wrong.
    """
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
        
        # Poll the IMU at roughly BEAMNG_IMU_FREQ (via BEAMNG_POLL_INTERVAL)
        # for BEAMNG_TEST_DURATION seconds, recording the absolute vertical
        # ('z') acceleration reading at each sample.
        while time.time() - start_time < BEAMNG_TEST_DURATION:
            sensor_data = imu_sensor.poll()
            if sensor_data and 'z' in sensor_data:
                vertical_g_forces.append(abs(sensor_data['z']))
            time.sleep(BEAMNG_POLL_INTERVAL)

        test_vehicle.ai.set_mode('disabled')
        beamng_client.close()

        # The single largest recorded vertical acceleration magnitude is the
        # real-physics analog of this app's `cost_score` (peak pitch
        # acceleration) -- both are "worst-moment-of-harshness" metrics,
        # though measured on different axes (linear G-force vs. angular
        # pitch acceleration), so they're comparable in trend but not
        # directly equal in units.
        peak_vertical_g = max(vertical_g_forces) if vertical_g_forces else 0.0
        return {"status": "Success", "peak_vertical_g": peak_vertical_g}

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 3. STREAMLIT USER INTERFACE DASHBOARD
# ==========================================
# Layout: a sidebar with 4 manual sliders (one per suspension parameter)
# plus a button to run the AI optimizer; a left column showing the live
# Python physics simulation for whatever sliders are currently set; a
# right column offering an optional in-game BeamNG validation run; and,
# once the optimizer has been run, a results section with a button to
# copy the AI's recommended settings back into the sliders.
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
    ax.set_ylabel("Heave from Static Ride Height (m)")
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
    
    # --- Step 1: Random initial exploration ---
    # Bayesian Optimization needs a handful of (parameters -> cost) data
    # points before it can fit a meaningful surrogate model, so we start by
    # sampling BO_NUM_INITIAL_SAMPLES points uniformly at random across the
    # full 4D search space and running the actual simulator on each one.
    # training_inputs has shape (BO_NUM_INITIAL_SAMPLES, 4); each row is one
    # (front_spring, front_damping, rear_spring, rear_damping) sample.
    training_inputs = np.random.uniform(
        low=[FRONT_SPRING_MIN, FRONT_DAMPING_MIN, REAR_SPRING_MIN, REAR_DAMPING_MIN], 
        high=[FRONT_SPRING_MAX, FRONT_DAMPING_MAX, REAR_SPRING_MAX, REAR_DAMPING_MAX], 
        size=(BO_NUM_INITIAL_SAMPLES, 4)
    )
    training_scores = np.array([simulator.simulate_bump(*sample)[0] for sample in training_inputs])
    
    @st.cache_resource
    def get_gaussian_process():
        # A Gaussian Process (GP) is a probabilistic model that, given a set
        # of observed (x, y) points, predicts not just an estimated y for
        # any new x, but also a calibrated uncertainty (std dev) around
        # that estimate -- higher uncertainty in regions far from any
        # observed sample, lower uncertainty near/between observed samples.
        # The Matern kernel controls how "smooth" the GP assumes the
        # underlying cost function is between samples; nu=1.5 is a
        # moderate-smoothness choice well suited to physical simulations
        # that can have some sharp-ish local structure.
        return GaussianProcessRegressor(
            kernel=Matern(nu=BO_MATERN_NU), 
            n_restarts_optimizer=BO_RESTARTS, 
            random_state=BO_RANDOM_STATE
        )

    gaussian_process = get_gaussian_process()
    
    best_score_record = float('inf')
    best_parameters_found = None
    
    # --- Step 2: Guided (Bayesian) search ---
    # Each iteration: (a) fit the GP to all data seen so far, (b) use the
    # GP to pick the most promising next point to test (without running the
    # expensive simulator), (c) actually run the simulator at that point,
    # and (d) add the new (params, cost) pair back into the training set so
    # the GP gets progressively more accurate near the optimum.
    for iteration_index in range(BO_TOTAL_ITERATIONS):
        # Fit (train) the GP surrogate model on all (parameters -> cost)
        # pairs observed so far. Fitting a GP means computing the
        # kernel-based covariance structure that lets it interpolate
        # between/extrapolate near these observed points.
        gaussian_process.fit(training_inputs, training_scores)
        
        def acquisition_function(candidate_params):
            """
            Lower Confidence Bound (LCB) acquisition function:
                LCB(x) = mu(x) - kappa * sigma(x)
            where mu(x) is the GP's predicted mean cost at x and sigma(x)
            is the GP's predicted uncertainty (standard deviation) at x.

            Minimizing LCB balances two competing goals:
              * Exploitation: mu(x) alone would push toward points already
                believed to have low cost.
              * Exploration: subtracting kappa * sigma(x) rewards trying
                points the GP is still very uncertain about, since they
                might secretly be even better than anything seen so far.
            BO_LCB_KAPPA (1.96) sets how strongly exploration is weighted.
            """
            reshaped_candidate = np.array(candidate_params).reshape(1, -1)
            predicted_mean, predicted_std = gaussian_process.predict(reshaped_candidate, return_std=True)
            return predicted_mean - BO_LCB_KAPPA * predicted_std

        # Minimize the (cheap, GP-based) acquisition function to decide
        # where to sample the (expensive, real) simulator next. L-BFGS-B is
        # a fast gradient-based local optimizer that respects the box
        # constraints in SEARCH_BOUNDS; starting from the fixed midpoint
        # guess each time is a simplification (a more thorough
        # implementation might use multiple random restarts here too).
        optimization_result = minimize(
            acquisition_function, BO_INITIAL_GUESS, bounds=SEARCH_BOUNDS, method='L-BFGS-B'
        )
        next_candidate_params = optimization_result.x
        
        # Actually run the physics simulator at the chosen candidate point
        # -- this is the one "expensive" evaluation per iteration.
        candidate_score, _, _ = simulator.simulate_bump(*next_candidate_params)
        
        # Append this new data point to the training set so future GP fits
        # take it into account (np.vstack stacks it as a new row; np.append
        # adds the matching cost value).
        training_inputs = np.vstack((training_inputs, next_candidate_params))
        training_scores = np.append(training_scores, candidate_score)
        
        # Track the best (lowest-cost) parameter set seen across every
        # sample so far, whether from the initial random phase or any
        # guided iteration.
        if candidate_score < best_score_record:
            best_score_record = candidate_score
            best_parameters_found = next_candidate_params
            
        progress_bar.progress((iteration_index + 1) / BO_TOTAL_ITERATIONS)
        status_text.text(f"Iteration {iteration_index+1}/{BO_TOTAL_ITERATIONS} completed. Best Score: {best_score_record:.4f}")
        time.sleep(0.1)  # brief pause so the progress bar visibly animates
        
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