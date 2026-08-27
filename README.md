# AI-Powered Suspension Tuning Advisor 🚗

A semi-realistic half-car (pitch-plane) suspension simulator paired with a Bayesian Optimization AI that searches for ride-comfort-optimal spring and damper settings — with optional validation against real physics via BeamNG.drive telemetry.

## Overview

This project models a vehicle as a **half-car pitch model**: a sprung chassis mass connected to independent front and rear unsprung wheel masses via spring-damper suspension units, which in turn contact the road through tire stiffness elements. The car is driven over a speed bump, and the simulator numerically integrates the equations of motion to see how the chassis pitches and bounces.

A **Gaussian Process Bayesian Optimization** loop then explores the 4-dimensional parameter space (front/rear spring rates and damping coefficients) to find the setup that minimizes peak pitch acceleration — a proxy for ride comfort/harshness.

Everything is wrapped in an interactive **Streamlit dashboard** so you can manually tune parameters, watch the simulated response, run the AI optimizer, and (if BeamNG.drive is installed) validate the winning setup in-game.

## Features

- **Physics-based half-car model** — 8-state ODE system (chassis heave/pitch + front/rear wheel hop) solved with `scipy.integrate.solve_ivp`, including tire lift-off protection and a road-input time lag between front and rear axles.
- **Interactive dashboard** — live sliders for spring rates and damping, real-time cost score and response plot.
- **AI optimization** — Gaussian Process surrogate model (Matérn kernel) + Lower Confidence Bound acquisition function, optimized each iteration with L-BFGS-B, to intelligently search for the best suspension setup in far fewer simulations than a grid search.
- **BeamNG.drive validation bridge** — optional integration that spawns a vehicle in BeamNG, drives it over a bump at a set speed using the in-game AI, and measures real peak vertical G-forces via an IMU sensor, so you can sanity-check the Python model against a full game physics engine.
- **Graceful degradation** — if `beamngpy` or the game itself isn't available, the app still runs fully in "simulation-only" mode.

## How It Works

1. **Simulate**: `HalfCarPitchSimulator` integrates the coupled equations of motion for a car passing over a 5 cm speed bump, returning a cost score equal to peak absolute pitch acceleration.
2. **Optimize**: The Bayesian Optimization loop starts with a handful of random samples, fits a Gaussian Process to the (parameters → cost) mapping, and iteratively proposes new parameter sets that balance exploring uncertain regions and exploiting known good regions.
3. **Validate** *(optional)*: The best (or manually chosen) parameters can be sent to BeamNG.drive, where an AI-driven vehicle runs the same test at the same speed, and real IMU data confirms whether the simulator's predictions hold up in a higher-fidelity engine.

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

## Tunable Parameters

| Parameter | Range | Description |
|---|---|---|
| Front Spring Rate | 20,000–150,000 N/m | Front suspension stiffness |
| Front Damping | 1,000–10,000 Ns/m | Front shock absorber damping |
| Rear Spring Rate | 20,000–150,000 N/m | Rear suspension stiffness |
| Rear Damping | 1,000–10,000 Ns/m | Rear shock absorber damping |

Vehicle mass, wheelbase, tire stiffness, bump geometry, and optimization settings (iterations, kernel, exploration weight, etc.) are all configurable at the top of `MAIN.py`.

## Notes on Realism

This is a **half-car** (2D, pitch-plane) model — it captures front/rear weight transfer and pitching motion but not roll (side-to-side) dynamics, which would require a full 3D quarter/full-car model. Tire behavior is simplified to a linear spring with lift-off protection (no lateral forces, no damping in the tire itself). It's a solid approximation for studying ride comfort over bumps, but not a substitute for a full vehicle dynamics simulation.
