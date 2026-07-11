from controller import NMPCController
import numpy as np
import mujoco
import mujoco.viewer
from filter import UnscentedKalmanFilter
from planner import RRTStarPlanner  # Import the new layer

def main():
    # 1. Setup Environment
    model = mujoco.MjModel.from_xml_path('3DoFarm.xml')
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    target_body_id = model.body('target').id
    obs_mocap_id = model.body('obstacle').mocapid[0]

    # Initialize modules
    controller = NMPCController(dt=0.02, horizon=20)
    ukf = UnscentedKalmanFilter(dt=0.02) 
    planner = RRTStarPlanner(safe_radius=0.38) # Initialize high-level planner

    tolerance = 0.03  
    target_pos = data.xpos[target_body_id] + np.array([0.0, 0.0, 0.07])
    
    swing_speed = 8.0  
    swing_distance = 0.25 
    u_prev = np.zeros(4)

    # --- UPGRADE POINT: GLOBAL TASK PLANNING STEP ---
    print("🤖 Computing High-Level Cartesian Path via RRT*...")
    # Acquire starting end-effector position using the kinematic framework
    init_q = data.qpos[:4].copy()
    start_ee_pos = np.array(controller.kin.forward_kinematics_sym(init_q)).flatten()
    init_obs_pos = data.mocap_pos[obs_mocap_id].copy()

    # Generate sparse path nodes, then interpolate into clean execution waypoints
    raw_path = planner.plan(start_ee_pos, target_pos, init_obs_pos)
    waypoints = planner.generate_waypoints(raw_path, horizon=20, num_points=200)
    print(f"✅ Path found! Generated {len(waypoints)} trajectory tracking waypoints.")

    trajectory_idx = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # --- 1. SENSOR DATA INJECTION ---
            raw_sensor_bus = np.array(data.sensordata)
            noisy_pos = raw_sensor_bus[0:4] + np.random.normal(0, 0.005, 4) 
            noisy_vel = raw_sensor_bus[4:8] + np.random.normal(0, 0.02, 4)
            z_meas = np.concatenate((noisy_pos, noisy_vel))

            # --- 2. STATE ESTIMATION (UKF) ---
            ukf.predict(u_prev)
            q_est, dq_est = ukf.update(z_meas)

            # --- 3. DYNAMIC OBSTACLE INJECTION ---
            sim_time = data.time
            data.mocap_pos[obs_mocap_id][1] = np.sin(sim_time * swing_speed) * swing_distance
            obs_pos = data.mocap_pos[obs_mocap_id].copy()

            # --- 4. EXTRACT HORIZON SLICE FOR NMPC ---
            # Slice a window of size N+1 future coordinates from our waypoint list
            horizon_trajectory = waypoints[trajectory_idx : trajectory_idx + controller.N + 1]

            # --- 5. SOLVE LOCAL NMPC ---
            try:
                # Pass the time-indexed vector window instead of a static target point
                optimal_acc = controller.solve(q_est, dq_est, horizon_trajectory, obs_pos)
            except Exception as e:
                print(f"Solver failed: {e}")
                optimal_acc = np.zeros(4)

            # Evaluate execution precision relative to current waypoint target
            ee_pos_est = np.array(controller.kin.forward_kinematics_sym(q_est)).flatten()
            distance_to_final = np.linalg.norm(ee_pos_est - target_pos)

            if distance_to_final < tolerance:
                print(f"🎯 Goal Reached! Terminal Error: {distance_to_final*1000:.1f} mm.")

            # Step our trajectory track forward if the arm progresses
            if trajectory_idx < len(waypoints) - (controller.N + 2):
                trajectory_idx += 1

            # --- 6. ACTUATION STEP ---
            u_prev = optimal_acc.copy()
            data.qacc[:4] = optimal_acc
            mujoco.mj_inverse(model, data)
            data.ctrl[:4] = data.qfrc_inverse[:4].copy()
            
            for _ in range(10):
                mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == '__main__':
    main()