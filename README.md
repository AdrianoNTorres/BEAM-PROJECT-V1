# AI-Powered Suspension Tuning Advisor 🚗

I built this to answer a question that's actually pretty annoying to answer by hand: *if I stiffen the front springs and soften the rear dampers, does the car ride better or worse?* Suspension tuning has four numbers that all interact with each other — front spring rate, front damping, rear spring rate, rear damping — and the "good" combination depends on what you're optimizing for (a plush ride vs. a planted, track-ready car). Rather than guessing and re-testing forever, this project simulates the car's physics and lets a Bayesian Optimization AI search the setting space for you, thousands of times faster than you could ever do by twisting real knobs.

At a high level, there are three moving pieces:

1. A **half-car pitch simulator** — a physics model of a car driving over a speed bump, solved with real differential equations.
2. A **Bayesian Optimization loop** — an AI search strategy that tries a handful of settings, builds a statistical model of "settings → ride quality," and uses that model to intelligently pick the next setting to try, instead of blindly guessing.
3. An **optional BeamNG.drive bridge** — if you own the driving simulator BeamNG.drive, the app can drive the winning setup over a real in-game bump and compare its physics engine's answer to the Python model's prediction, as a sanity check.

Everything lives in an interactive **Streamlit dashboard**, so you can drag sliders and watch the car's response update live, or hit a button and let the optimizer do the work.

---

## Part 1 — What this actually does

### The car, simplified

Modeling every nut and bolt of a real suspension would be its own PhD. Instead, this uses a well-established simplification from vehicle dynamics called a **half-car pitch model**. Picture looking at the car from the side, like a silhouette: you can see it bounce up and down, and you can see its nose dip and rise (pitch), but you can't see anything about it leaning left-right (roll), because that's invisible from the side. That's exactly the slice of reality this model captures — bounce and pitch, not roll — which is the right trade-off for asking "how does the setup feel when you drive straight over a bump," while being far cheaper to compute than a full 3D vehicle model.

The car is represented as four masses connected by springs and dampers:

- The **chassis** (the sprung mass — everything sitting "above" the suspension: body, engine, passengers)
- The **front wheel/tire assembly** and **rear wheel/tire assembly** (the unsprung masses — the parts that move with the road)

The chassis connects to each wheel through a spring-damper unit (that's the actual suspension — the spring rate and damping you're tuning), and each wheel connects to the road through the tire, which acts like its own, much stiffer spring.

### The test: driving over a bump

The simulated car starts at rest, settled into its natural ride height under its own weight (its **static equilibrium** — more on that below), then drives over a smooth speed bump. The software tracks, at every instant in time, how much the chassis bounces, how much it pitches, and how the front and rear wheels move — the same way an engineer reading accelerometer traces from a real test car would.

### Turning "how it rode" into one number

A time-series of wiggly lines isn't something an algorithm can directly optimize — it needs a single number to minimize. So the simulator distills the whole bump event into a **cost score**: a weighted blend of things like peak pitch jolt, how long the car keeps bouncing after the bump, and how consistently the tires stay loaded against the road. Lower is smoother/better-controlled. What exactly counts as "better" is adjustable — see the *Ride Profiles* discussion below — because a grand-touring sedan and a track car don't actually want the same thing.

### The AI search

Once there's a single cost number for any given (front spring, front damping, rear spring, rear damping) combination, the problem becomes: *find the four numbers that make that cost as low as possible.* A brute-force grid search over four continuous dimensions would need a huge number of simulations to cover the space adequately. Instead, this uses **Bayesian Optimization**: the AI runs a small batch of random settings first, fits a statistical model of how cost varies across the setting space, and then repeatedly asks "given what I know so far, where should I try next?" — balancing trying promising-looking regions against exploring regions it's still uncertain about. This routinely finds a near-optimal setup in a few dozen simulations instead of thousands.

### Checking the model against reality (optional)

The physics model here, like any simplified model, is an approximation. If you have BeamNG.drive — a physics-driven driving simulation game — installed, the app can hand the winning setup to an AI driver in-game, run it over a real (in-engine) bump at speed, and read back real IMU-style motion data. That lets you sanity-check whether the simplified math model's predictions actually hold up against a much higher-fidelity physics engine. If you don't have the game, none of this is required — the app runs perfectly well in simulation-only mode.

---

## Part 2 — The math behind it (for the curious / technically inclined)

*This section lays out the actual equations of motion, the numerical methods, and the cost function, the way you'd find them written up in a vehicle-dynamics paper. Skip it if you just want to use the dashboard — read it if you want to know exactly what's happening under the hood, or if you want to modify the model yourself.*

### 2.1 System definition

The half-car has four degrees of freedom, giving an 8-dimensional state vector once each is paired with its velocity:

$$
\mathbf{x} = \begin{bmatrix} z_s & \dot z_s & \theta & \dot\theta & z_{uf} & \dot z_{uf} & z_{ur} & \dot z_{ur} \end{bmatrix}^T
$$

where $z_s$ is chassis heave (vertical displacement of the center of mass), $\theta$ is chassis pitch angle, and $z_{uf}, z_{ur}$ are the front and rear unsprung (wheel) vertical displacements.

**Parameters used in the model:**

| Symbol | Meaning | Value |
|---|---|---|
| $m_s$ | Sprung (chassis) mass | 1455 kg |
| $m_{uf}, m_{ur}$ | Front/rear unsprung mass | 40 kg each |
| $I_\theta$ | Pitch moment of inertia | $m_s \cdot a \cdot b$ (a lumped-mass approximation) |
| $a$ | Distance, CG → front axle | 1.3958 m |
| $b$ | Distance, CG → rear axle | 1.4498 m |
| $k_t$ | Tire vertical stiffness | 200,000 N/m |
| $c_t$ | Tire damping | 2,500 Ns/m |
| $g$ | Gravity | 9.81 m/s² |

The four tunable parameters — front/rear spring rate ($k_f$, $k_r$) and front/rear damping ($c_f$, $c_r$) — are what the optimizer searches over.

### 2.2 Static equilibrium (the starting point)

Before the bump, the car sits at rest under gravity. That resting state — ride height and any static pitch — is found by solving the nonlinear system for zero net force and torque on every mass:

$$
F_f + F_r - m_s g = 0, \qquad F_r b - F_f a = 0
$$

$$
F_{t,f} - F_f - m_{uf} g = 0, \qquad F_{t,r} - F_r - m_{ur} g = 0
$$

where $F_f, F_r$ are the front/rear suspension (spring) forces and $F_{t,f}, F_{t,r}$ are the tire contact forces. This is solved numerically with a nonlinear root-finder (`scipy.optimize.root`), and the resulting displacements become the initial condition — otherwise the simulation would open with the chassis "falling" onto its springs before any bump even happens, contaminating the response with a spurious transient.

### 2.3 Equations of motion

At any instant, suspension **compression** is the relative displacement between the wheel and the chassis point directly above it:

$$
\delta_f = z_{uf} - (z_s - a\theta), \qquad \delta_r = z_{ur} - (z_s + b\theta)
$$

and the corresponding suspension forces follow a standard linear spring-damper law:

$$
F_f = k_f \delta_f + c_f \dot\delta_f, \qquad F_r = k_r \delta_r + c_r \dot\delta_r
$$

The tire is modeled the same way, but against the road profile $z_{road}(t)$ instead of the chassis, and is clamped at zero rather than allowed to go negative — a tire can push against the road but can't pull the wheel down (no downward "suction"), which is what real tire lift-off looks like:

$$
F_{t} = \max\!\big(0,\ k_t(z_{road} - z_u) + c_t(\dot z_{road} - \dot z_u)\big)
$$

Newton's second law then gives the four accelerations that actually get integrated:

$$
\ddot z_s = \frac{F_f + F_r}{m_s} - g
$$

$$
\ddot\theta = \frac{F_r b - F_f a}{I_\theta}
$$

$$
\ddot z_{uf} = \frac{F_{t,f} - F_f}{m_{uf}} - g, \qquad \ddot z_{ur} = \frac{F_{t,r} - F_r}{m_{ur}} - g
$$

This 8-state ODE system is integrated forward in time with `scipy.integrate.solve_ivp` (an adaptive-step Runge-Kutta method), capped at a maximum step size of a quarter of the bump's duration so the fast bump transient is never accidentally stepped over.

One subtlety worth calling out: the rear axle doesn't hit the bump at the same instant as the front — it's delayed by wheelbase ÷ speed, a **road-input time lag** ($\Delta t_{axle}$) baked into the road-profile function so the pitch dynamics (front rising while rear is still flat, and vice versa) come out physically correct instead of both axles bouncing in lockstep.

### 2.4 The cost function

The optimizer needs one scalar to minimize, so every simulated bump event is compressed into a weighted sum of **dimensionless** ride-quality terms — dimensionless because each raw quantity (in meters, rad/s², etc.) is first divided by a reference value of "typical" magnitude, so that a peak-acceleration term and a heave-displacement term can be added together meaningfully despite living in completely different units:

$$
J = \sum_i w_i \cdot \frac{q_i}{q_{i,\text{ref}}} \;+\; P_{\text{realism}}
$$

The terms $q_i$ that make up the sum are extracted from the simulated time-series:

- **Peak pitch acceleration** — $\max|\ddot\theta|$ during the bump event: the single sharpest jolt.
- **RMS pitch acceleration** — root-mean-square of $\ddot\theta$ across the bump event, since a peak alone can miss a ride that's uniformly harsh without one dramatic spike.
- **Post-bump heave RMS, pitch-velocity RMS, and heave-acceleration RMS** — how much the car keeps moving *after* the bump has passed, i.e. how well the suspension settles down rather than continuing to wallow.
- **Post-bump heave peak-to-peak** — the total swing of the settling oscillation.
- **Oscillation persistence** — the post-bump window is split in half, and late-window heave RMS is compared to early-window heave RMS. A ratio near (or above) 1 means the car is still shaking just as hard well after the bump — "a ride that never settles" — a failure mode that a single RMS or peak-to-peak number averaged over the whole window can quietly miss, since a sharp-but-short shake and a low-amplitude-but-endless one can land on similar RMS values despite feeling very different to a passenger.
- **Dynamic tire load variation** — RMS deviation of tire contact force from its static value during the bump, which matters most for grip (a tire that's momentarily unloaded can't corner or brake).
- **Spring rate sum** — a mild penalty on stiffness, so "just make everything infinitely stiff" doesn't win by accident.

Each term's weight $w_i$ is set by a **ride profile** (Comfort, Balanced, Track, or a fully custom weighting), which is what lets the same simulator and optimizer answer "what's the smoothest possible ride" and "what's the most controlled, grippiest setup" as two different questions with two different answers.

**Realism penalties**, $P_{\text{realism}}$, exist to stop the optimizer from gaming the objective with physically silly setups. Two are used:

*Underspring penalty* — discourages near-zero spring rates (which can look artificially good on a pure ride-response score) by penalizing the fractional shortfall below a minimum spring rate floor $k_{min}$:

$$
P_{spring} = \left(\frac{\max(0,\, k_{min}-k_f)}{k_{min}}\right)^{2} + \left(\frac{\max(0,\, k_{min}-k_r)}{k_{min}}\right)^{2}
$$

*Damping-ratio penalty* — keeps the damping physically sensible relative to each spring rate, using the standard dimensionless damping ratio $\zeta = c / c_{crit}$, where critical damping is $c_{crit} = 2\sqrt{k \cdot m}$ (using each axle's share of sprung mass, $m_f = m_s b/L$ and $m_r = m_s a/L$ for wheelbase $L = a+b$). Deviation beyond a tolerance band around a target ratio $\zeta_{target}$ is penalized quadratically, with the "tolerance-widths past the band" capped before squaring, so one extreme outlier setup can't blow up and swamp every other term in the sum.

### 2.5 The optimization loop

With $J(k_f, c_f, k_r, c_r)$ defined as a black-box function (only evaluable by actually running the simulator — no closed form), the search uses **Gaussian Process (GP) Bayesian Optimization**:

1. **Initial exploration** — a handful of settings are sampled uniformly at random across the search space and evaluated, to seed the model with some real data.
2. **Surrogate modeling** — a Gaussian Process regressor with a **Matérn kernel** is fit to the (settings → cost) data collected so far. The GP doesn't just predict a cost for any untested setting — it predicts a mean *and* a confidence interval, i.e. "here's my best guess, and here's how sure I am."
3. **Acquisition** — the next setting to test is chosen by minimizing a **Lower Confidence Bound (LCB)** acquisition function,

$$
\alpha(\mathbf{p}) = \mu(\mathbf{p}) - \kappa\,\sigma(\mathbf{p})
$$

&nbsp;&nbsp;&nbsp;&nbsp;where $\mu$ is the GP's predicted cost, $\sigma$ is its uncertainty, and $\kappa$ controls the explore/exploit trade-off — a larger $\kappa$ pushes the search toward uncertain regions (exploration) rather than purely toward the current best-known region (exploitation). This is itself a small optimization problem, solved with multiple random-restarted L-BFGS-B runs so it doesn't get stuck in a local minimum of the acquisition surface.
4. **Repeat** — the newly tested point is added to the training data, the GP is refit, and the loop continues, with early stopping once a run of iterations fails to meaningfully improve on the best score found so far.

This is dramatically more sample-efficient than a grid search precisely because the GP's uncertainty estimate tells the search where more information is actually valuable, instead of spending simulations on settings a competent model already expects to be mediocre.

### 2.6 What the dashboard shows you

The Streamlit app turns all of the above into live charts: the chassis heave/pitch **time evolution** as the car crosses the bump (so you can watch the transient and the settling behavior directly, not just read a single cost number), and a running plot of the optimizer's score-per-trial alongside its best-score-so-far curve, so you can literally watch the AI convergence happen in real time.

---

## Requirements

```
numpy
scipy
matplotlib
streamlit
scikit-learn
beamngpy   # optional — only needed for in-game validation
```

BeamNG.drive itself (the game) must be installed and running locally for the validation feature to work; without it, the app simply reports that validation is unavailable and continues to function normally.

## Usage

```bash
streamlit run MAIN.py
```

Then in the dashboard:
1. Adjust the front/rear spring rate and damping sliders to explore the physics model manually.
2. Click **"Run Bayesian Optimization Loop"** to let the AI search for the ride-comfort-optimal setup.
3. Click **"Apply Optimal Parameters to Sliders"** to load the AI's recommendation.
4. (Optional) Click **"Run Automated BeamNG Test Lap"** to validate the current settings in-game.

## Tunable parameters

| Parameter | Range | Description |
|---|---|---|
| Front Spring Rate | 20,000–150,000 N/m | Front suspension stiffness |
| Front Damping | 1,000–10,000 Ns/m | Front shock absorber damping |
| Rear Spring Rate | 20,000–150,000 N/m | Rear suspension stiffness |
| Rear Damping | 1,000–10,000 Ns/m | Rear shock absorber damping |

Vehicle mass, wheelbase, tire stiffness, bump geometry, and optimization settings (iterations, kernel, exploration weight, etc.) are all configurable at the top of `MAIN.py`.

## Notes on realism

This is a **half-car** (2D, pitch-plane) model — it captures front/rear weight transfer and pitching motion but not roll (side-to-side) dynamics, which would require a full 3D quarter/full-car model. Tire behavior is simplified to a linear spring with lift-off protection (no lateral forces, no damping in the tire itself). It's a solid approximation for studying ride comfort over bumps, but not a substitute for a full vehicle dynamics simulation.