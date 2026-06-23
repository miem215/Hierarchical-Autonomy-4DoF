## Project Overview

This project simulates a 4 Degree-of-Freedom (4-DOF) robotic arm in the MuJoCo physics engine. It utilizes a custom-built **Nonlinear Model Predictive Controller (NMPC)** powered by CasADi to calculate optimal trajectories, coupled with an **Unscented Kalman Filter (UKF)** to handle noisy real-world sensor data.

The system is designed to track a target coordinate while proactively dodging a dynamic, swinging obstacle, demonstrating advanced optimal control and state estimation in a simulated hardware environment.

## Motivation & Project Evolution

This project evolved through several modifications to arrive current control architecture:

* **Kinematic Upgrade (3-DOF to 4-DOF):** The manipulator was upgraded from 3 to 4 degrees of freedom. This added kinematic redundancy allows the arm to maintain a positional lock on the target while simultaneously contorting its internal posture to dodge obstacles.
* **Dynamic Environments:** The project transitioned from using stationary obstacle to dynamic obstacle. This necessitated the shift to real-time Nonlinear Model Predictive Control (NMPC) to proactively predict and recalculate safe trajectories on the fly.
* **State Estimation:** To simulate physical hardware limitations, pristine simulation data was intentionally corrupted with Gaussian noise. An Unscented Kalman Filter (UKF) was implemented from scratch to clean this sensor data before feeding it to the NMPC, mimicking the exact pipeline required for real-world hardware encoders.
* **Whole-Body Collision Awareness:** The collision constraints evolved from simple "end-effector-only" checking to a planar whole-body force field. Virtual nodes are now mathematically calculated along the arm's links to prevent intermediate joints (like the elbow or wrist) from clipping the obstacle.

<img width="589" height="501" alt="4DoF_arm" src="https://github.com/user-attachments/assets/bb05b217-597d-4e91-8c6b-935fcfb1d968" />

## Design Justification & Computational Profiling

To ensure this controller is viable for physical hardware deployment, specific algorithmic trade-offs were made to prioritize **real-time execution (50Hz)**:

* **Explicit Euler Dynamics:** Explicit Euler integration was chosen over higher-order solvers like Runge-Kutta 4 (RK4). While RK4 offers superior prediction accuracy, Explicit Euler guarantees a strict sub-20ms solve time, which is critical for closing the control loop in real-time.
* **Predictive Horizon ($N=20$):** At a timestep of $\Delta t = 0.02s$, a 20-step horizon yields a 0.4-second predictive window. This provides just enough spatial awareness for the IPOPT solver to dodge dynamic obstacles without causing computational bottlenecks.
* **Solve Time Performance:** The CasADi IPOPT solver successfully completes the 20-step non-linear horizon in approximately 10-15 milliseconds on an average CPU, running comfortably within the 20ms allowance required for stable 50Hz control.

## Sim-to-Real Considerations

* **Sensor Noise Injection:** Gaussian noise ($\mathcal{N}(0, R)$) is continuously injected into MuJoCo's pristine joint position and velocity sensor buses to simulate encoder inaccuracies.
* **Why the UKF?** An Unscented Kalman Filter (UKF) was implemented from scratch to clean this noisy data before it reaches the NMPC. In the current architecture, the state vector is $x = [q, \dot{q}]^T$. Because we use a simple one-step Euler integration for the process model, and because the measurements are direct simulated encoder readings (mapping 1:1 with the states), the entire system is strictly linear:

**Current Linear Process Model (Explicit Euler):**
$$
x_{k+1} = \\begin{bmatrix} q_k + \dot{q}_k \Delta t \\\\ \dot{q}_k + u_k \Delta t \\end{bmatrix} + w_k
$$

**Current Linear Measurement Model (Encoders):**
$$
z_k = I \\cdot x_k + v_k
$$

Because both models are linear, a standard Linear Kalman Filter (KF) would technically suffice. However, the UKF was explicitly chosen as an **architectural future-proofing** measure. To simulate real-world lab conditions in future iterations, Cartesian camera data $(x, y, z)$ will be fused with the encoder data.

**Future Non-Linear Measurement Model (Camera Fusion):**
$$
z_{cam} = \\text{FK}(q_k) + v_{cam}
$$

Because the Forward Kinematics ($\\text{FK}$) relies on highly non-linear trigonometric transformations, a Linear KF will fail. The UKF's deterministic sigma points are already in place to naturally handle this future non-linear measurement update without requiring a complete estimator rewrite or complex Jacobian derivations.


## Mathematical Formulation

### 1. System Dynamics

The NMPC predicts the future states of the arm over a horizon $N$ using Explicit Euler integration. Let the state vector be $x = [q, \dot{q}]^T \\in \\mathbb{R}^8$ and the control input be joint accelerations $u = \ddot{q} \\in \\mathbb{R}^4$. The system dynamics are defined as:

$$
x_{k+1} = \\begin{bmatrix} q_{k+1} \\\\ \dot{q}_{k+1} \\end{bmatrix} = \\begin{bmatrix} q_k + \dot{q}_k \Delta t \\\\ \dot{q}_k + u_k \Delta t \\end{bmatrix}
$$

### 2. NMPC Cost Function

The CasADi solver minimizes a highly tuned cost function $J$ across the prediction horizon. The cost function balances aggressive target tracking with energy efficiency and postural stability:

$$
J = \sum_{k=0}^{N-1} \left( J_{track, k} + J_{effort, k} + J_{posture, k} + J_{slack, k} \right) + J_{terminal}
$$

Where the individual running costs are defined as:

* **Target Tracking:** $J_{track, k} = 500 \\| \\text{FK}(q_k) - p_{target} \\|_2^2$
* **Control & Velocity Effort:** $J_{effort, k} = 0.2 \\| u_k \\|_2^2 + 0.2 \\| \dot{q}_k \\|_2^2$
* **Postural Alignment:** $J_{posture, k} = (q_k - q_{home})^T W_{posture} (q_k - q_{home})$
* **Obstacle Slack Penalty:** $J_{slack, k} = W_{obs} \cdot s_k$ (where $W_{obs} = 100,000$)

### 3. Whole-Body Collision Avoidance (Virtual Nodes)

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

## Algorithmic Limitations & Local Minima

Because the collision avoidance relies on a soft-constraint slack variable formulation, the resulting optimization landscape is highly **non-convex**. 

* **The Local Minimum Trap:** If the dynamic obstacle swings perfectly along the line-of-sight between the end-effector and the target, the IPOPT solver can occasionally fall into a local minimum. To navigate *around* the obstacle, the NMPC must temporarily move away from the target (increasing the immediate tracking cost). 
* **Future Mitigations:** Future work will explore integrating a higher-level global path planner (such as RRT* or A*) to provide collision-free waypoints, or implementing Control Barrier Functions (CBFs) to mathematically force the solver out of these non-convex traps.

## File Structure

* `main.py` - The core simulation loop. Initializes MuJoCo, injects Gaussian noise, and ties the UKF and NMPC pipelines together.
* `controller.py` - Contains the CasADi optimization logic. Defines the explicit dynamics, cost functions, and non-linear collision constraints.
* `filter.py` - Contains the Unscented Kalman Filter (UKF) implementation. 
* `Kinematic.py` - The kinematic engine handling symbolic forward kinematics for the CasADi solver.
* `3DoFarm.xml` - The MuJoCo environment specification.
