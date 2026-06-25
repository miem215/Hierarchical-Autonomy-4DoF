## Project Overview

This project simulates a 4 Degree-of-Freedom (4-DOF) robotic arm in the MuJoCo physics engine. It utilizes a custom-built **Nonlinear Model Predictive Controller (NMPC)** powered by CasADi to calculate optimal trajectories, coupled with an **Unscented Kalman Filter (UKF)** to handle noisy real-world sensor data.

The system is designed to track a target coordinate while proactively dodging a dynamic, swinging obstacle, demonstrating advanced optimal control and state estimation in a simulated hardware environment.

## File Structure

* main.py - The core simulation loop. Initializes MuJoCo, injects Gaussian noise into the sensor bus, and ties the UKF and NMPC pipelines together.

* controller.py - Contains the CasADi optimization logic. Defines the explicit dynamics, cost functions, and non-linear collision constraints over a finite prediction horizon.

* filter.py - Contains the Unscented Kalman Filter (UKF) implementation. Uses deterministic sigma points to filter noisy joint positions and velocities.

* Kinematic.py - The kinematic engine handling symbolic forward kinematics for the CasADi solver.

* 3DoFarm.xml - The MuJoCo environment specification, defining the physical attributes of the arm, the dynamic obstacle, and the target.

## Motivation & Project Evolution

This project evolved through several modifications to arrive current control architecture:

* **Kinematic Upgrade (3-DOF to 4-DOF):** The manipulator was upgraded from 3 to 4 degrees of freedom. This added kinematic redundancy allows the arm to maintain a positional lock on the target while simultaneously contorting its internal posture to dodge obstacles.
* **Dynamic Environments:** The project transitioned from using stationary obstacle to dynamic obstacle. This necessitated the shift to real-time Nonlinear Model Predictive Control (NMPC) to proactively predict and recalculate safe trajectories on the fly.

<img width="751" height="626" alt="Animation" src="https://github.com/user-attachments/assets/c48e2b02-8a60-46db-8d11-46ad8244e542" />


## Why UKF?

* **Sensor Noise Injection:** Gaussian noise ($\mathcal{N}(0, R)$) is continuously injected into MuJoCo's pristine joint position and velocity sensor readings to simulate encoder inaccuracies.
* **Why the UKF?** An Unscented Kalman Filter (UKF) was implemented to clean this noisy data before it reaches the NMPC. In the current architecture, the state vector is $x = [q, \dot{q}]^T$. Because we use a simple one-step Euler integration for the process model, and because the measurements are direct simulated encoder readings (mapping 1:1 with the states), the entire system is strictly linear:

**Current Linear Process Model (Explicit Euler):**

$$
x_{k+1} = \begin{bmatrix} q_k + \dot{q}_k \Delta t \\ \dot{q}_k + u_k \Delta t \end{bmatrix} + w_k
$$

**Current Linear Measurement Model (Encoders):**

$$
z_k = I \cdot x_k + v_k
$$

Because both models are linear, a standard Linear Kalman Filter (KF) would technically suffice. However, the UKF was explicitly chosen as an **architectural future-proofing** measure. To simulate real-world lab conditions in future iterations, Cartesian camera data $(x, y, z)$ will be fused with the encoder data. 

**Future Non-Linear Measurement Model (Camera Fusion):**

$$
z_{cam} = \text{FK}(q_k) + v_{cam}
$$

Because the Forward Kinematics ($\text{FK}$) relies on highly non-linear trigonometric transformations, a Linear KF will fail. The UKF's deterministic sigma points are already in place to naturally handle this future non-linear measurement update without requiring a complete estimator rewrite or complex Jacobian derivations.

UKF performace on the joint velocity in current setup: 

<img width="3000" height="1500" alt="ukf_performance" src="https://github.com/user-attachments/assets/86baae36-f166-4b41-bcac-e079fb32501d" />

## Nonlinear Model Predictive Controller (NMPC)
### 1. Why NMPC

While the complete optimization problem balances multiple operational objectives (such as control effort and postural alignment), the core optimization landscape is fundamentally dominated by a "tug-of-war" between two primary costs: Target Tracking and Obstacle Avoidance.

Figure 1 visualizes this cost landscape mapped across the 2D Cartesian workspace of the end-effector. As demonstrated by the bifurcated topology, the addition of the obstacle avoidance penalty transforms the otherwise simple quadratic tracking problem into a highly non-convex optimization space.

This geometric reality is the fundamental justification for utilizing a Nonlinear Model Predictive Controller (NMPC). Standard Linear MPC architectures rely on Quadratic Programming (QP) solvers that require a strictly convex feasible region. If a QP solver were presented with this landscape, it would become trapped in a local minimum against the obstacle's boundary and fail to find a solution. By deploying an NMPC, the system leverages CasADi's advanced IPOPT interior-point solver, which is natively capable of evaluating non-linear trigonometric kinematics and routing trajectories around non-convex constraint manifolds.

To generate this geometric visualization, the simplified cost function $J$ is evaluated as:

$$
J = \sum_{k=0}^{N-1} \left( J_{track, k} + J_{slack, k} \right)
$$

* **Target Tracking:** $J_{track, k} = w_{track} \| p_{end_effecor} - p_{target} \|_2^2$
* **Obstacle Slack Penalty:** $J_{slack, k} = W_{obs} \cdot s_k$ (where $W_{obs} = 100,000$)

<img width="2245" height="2373" alt="optimization_landscape" src="https://github.com/user-attachments/assets/17f24016-833b-4487-85ba-f37dbf04749f" />


### 2. System Dynamics

The NMPC predicts the future states of the arm over a horizon $N$ using Explicit Euler integration. Let the state vector be $x = [q, \dot{q}]^T \in \mathbb{R}^8$ and the control input be joint accelerations $u = \ddot{q} \in \mathbb{R}^4$. The system dynamics are defined as:

$$
x_{k+1} = \begin{bmatrix} q_{k+1} \\ \dot{q}_{k+1} \end{bmatrix} = \begin{bmatrix} q_k + \dot{q}_k \Delta t \\ \dot{q}_k + u_k \Delta t \end{bmatrix}
$$

### 3. Cost Function

The CasADi solver minimizes a highly tuned cost function $J$ across the prediction horizon. The cost function balances target tracking, obstacle avoidance and energy efficiency and postural stability:

$$
J = \sum_{k=0}^{N-1} \left( J_{track, k} + J_{effort, k} + J_{posture, k} + J_{slack, k} \right) + J_{terminal}
$$

Where the individual running costs are defined as:

* **Target Tracking:** $J_{track, k} = 500 \| \text{FK}(q_k) - p_{target} \|_2^2$
* **Control & Velocity Effort:** $J_{effort, k} = 0.2 \| u_k \|_2^2 + 0.2 \| \dot{q}_k \|_2^2$
* **Postural Alignment:** $J_{posture, k} = (q_k - q_{home})^T W_{posture} (q_k - q_{home})$
* **Obstacle Slack Penalty:** $J_{slack, k} = W_{obs} \cdot s_k$ (where $W_{obs} = 100,000$)

## Whole-Body Collision Avoidance (Virtual Nodes)

To prevent the intermediate links from clipping through the dynamic obstacle, the arm calculates fast 2D planar kinematics (treating the obstacle as an infinite pillar along the Z-axis). For each joint/node, the radial distance $r$ in the X-Y plane is derived:

$$
r_{elbow} = L_2 \sin(q_2)
$$

$$
r_{wrist} = r_{elbow} + L_3 \sin(q_2 + q_3)
$$

A soft constraint is applied to the End-Effector, Wrist, Elbow, and interpolated Link Midpoints. A slack variable $s_k \geq 0$ allows the solver to find mathematically feasible routes if trapped:

$$
(x_{node} - x_{obs})^2 + (y_{node} - y_{obs})^2 + s_k \geq r_{safe}^2
$$

## Other design parameters

To ensure this controller is viable for physical hardware deployment, specific algorithmic trade-offs were made to prioritize **real-time execution (50Hz)**:

* **Explicit Euler Dynamics:** Explicit Euler integration was chosen over higher-order solvers like Runge-Kutta 4 (RK4). While RK4 offers superior prediction accuracy, Explicit Euler guarantees a strict sub-20ms solve time, which is critical for closing the control loop in real-time.
* **Predictive Horizon ($N=20$):** At a timestep of $\Delta t = 0.02s$, a 20-step horizon yields a 0.4-second predictive window. This provides just enough spatial awareness for the IPOPT solver to dodge dynamic obstacles without causing computational bottlenecks.
* **Solve Time Performance:** The CasADi IPOPT solver successfully completes the 20-step non-linear horizon in approximately 10-15 milliseconds on an average CPU, running comfortably within the 20ms allowance required for stable 50Hz control.

## Current State & Features

Advanced State Estimation: Filters Gaussian noise from joint sensors before feeding states into the controller.

Dynamic Postural Costs (NMPC): State-dependent cost weights. Distal joints stiffen when reaching from afar to act like a spear, and loosen dynamically as the end-effector enters the target zone.

Dynamic Obstacle Avoidance: Environment features a moving dynamic obstacle. The NMPC recalculates on the fly to dodge it.

Target Tracking & Hold: The arm aggressively pursues the target coordinate and switches to a stable hold/hover state upon breaching the tolerance threshold.

## Known Issues

Whole-Body vs. Tip Collision: Currently, the end-effector dodges the dynamic obstacle perfectly using a 2D planar force field constraint. However, the system struggles with strict Whole-Body Collision Avoidance. While intermediate virtual nodes have been drafted, the intermediate links can still occasionally clip the obstacle. The discrete virtual node approach requires further tuning to create a truly impenetrable force field along the 1-meter link lengths.

## Future Roadmap

Continuous Collision Avoidance: Replace the discrete "Virtual Node" point-mass constraints with true Line-Segment to Point distance formulas.

Computer Vision Integration: Replace the raw MuJoCo mocap_pos data with a simulated RGB camera pipeline to estimate the obstacle's state dynamically.

Dynamic Target Interception: Feed an estimated target velocity vector into the CasADi prediction horizon to intercept moving targets.

Reinforcement Learning Benchmarking: Wrap the environment in a Gymnasium interface to benchmark this NMPC's performance against PPO/SAC deep learning agents.
```bash
python main.py
```
