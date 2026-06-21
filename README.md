# 3-DOF NMPC & State Estimation in MuJoCo

> **Note on Implementation:** The project relies on NumPy for matrix operations and CasADi for symbolic optimization, interfacing directly with the MuJoCo physics engine. 

## Executive Summary
This repository demonstrates a fully custom, white-box robotics pipeline capable of controlling a 3-DOF manipulator to navigate spatial constraints and reach precise 3D Cartesian coordinates. The system bridges non-convex mathematical optimization with physical simulation through a robust discrete state machine.

**Visual Proof of Trajectory Optimization:**
*(Insert a side-by-side GIF here: Left side showing the greedy stage-cost failing/hitting the obstacle. Right side showing the sweeping terminal-cost trajectory successfully navigating the pillar.)*

---

## 1. Mathematical Formulation & Advanced Control

### Kinematics & Dynamics
The symbolic forward kinematics were derived custom to perfectly map the MuJoCo XML physical plant to the mathematical solver. System dynamics are propagated through the CasADi graph using custom Euler integration.

### Overcoming the 3-DOF Kinematic Limit
A fundamental challenge of a 3-DOF manipulator operating in 3D task space ($m=3$, $n=3$) is the lack of a null-space ($n - m = 0$). The arm cannot perform internal motions to dodge obstacles while maintaining the end-effector trajectory, forcing highly non-convex trajectory planning.

### Stage Cost vs. Terminal Cost Formulation
To bypass the greedy nature of standard NMPC stage costs—which penalize the temporary distance increases required to clear obstacles—the solver was refactored into a **Terminal Cost Formulation**. By shifting the primary tracking weight to the final state ($N$), the solver is mathematically incentivized to execute strategic, wide-arcing "extend and fold" trajectories to bypass spatial constraints.

### Soft Spatial Constraints
Physical obstacles are translated into mathematical 2D "Keep-Out Cylinders". To prevent the IPOPT solver from crashing due to infeasible initialization states or high-speed kinematic limits, hard spatial boundaries were relaxed using mathematical slack variables ($s_k$).

The optimization problem is formulated as:

$$\min_{U, s} \left( \sum_{k=0}^{N-1} \left( \|u_k\|_R^2 + W_{slack} s_k \right) \right) + \|x_N - x_{target}\|_Q^2$$

**Subject to:**
* **System Dynamics:** $x_{k+1} = f(x_k, u_k)$
* **Initial State:** $x_0 = x_{curr}$
* **Soft Spatial Constraint:** $(x_{ee} - x_{obs})^2 + (y_{ee} - y_{obs})^2 + s_k \ge r_{safe}^2$
* **Slack Positivity:** $s_k \ge 0$

---

## 2. State Machine & Systems Engineering
To bridge the gap between abstract optimization and hardware-ready reliability, the architecture utilizes a discrete state machine:

* **Active Tracking (NMPC Mode):** The CasADi solver continuously calculates the optimal acceleration over the prediction horizon.
* **Gravity-Compensated Hold Mode:** Upon entering a predefined $L_2$ norm tolerance zone (e.g., 20mm), the NMPC smoothly hands off control. Inverse dynamics compute the exact torque required to freeze the arm in place, commanding zero acceleration.
* **Robust Failsafes:** If the gradient-based solver detects an infeasible problem and fails, the system intercepts the error and defaults to a zero-torque command, preventing the retention of massive, unbounded acceleration buffers that damage physical hardware.

---

## 3. Sensor Noise & State Estimation (UKF)
*To be completed: This section will detail the injection of synthetic Gaussian noise to simulate real-world hardware encoders, and the implementation of an Unscented Kalman Filter (UKF) to close the loop.*

---

## 4. Applications in Complex Research Domains
The control strategies implemented in this architecture serve as direct foundational logic for advanced robotics research:

* **Underwater Robotics & Marine Biodiversity:** The soft mathematical constraint formulation ("Keep-Out Cylinders") is directly applicable to ROV manipulator control, ensuring robotic arms can operate near delicate, fragile reef structures without catastrophic collisions or mathematical solver crashes.
* **Interactive Alignment & Human-Robot Interaction (HRI):** The energy-penalized control effort ($R$) prevents violent, "bang-bang" trajectory planning. This smooth, predictable NMPC behavior is a strict prerequisite for safe deployment in shared human-robot workspaces.

---

## 5. Setup & Reproducibility

### Dependencies
* Python 3.10+
* `mujoco`
* `casadi`
* `numpy`

### Execution
Run the main pipeline to launch the passive viewer and initiate the optimal control sequence:
```bash
python main.py
