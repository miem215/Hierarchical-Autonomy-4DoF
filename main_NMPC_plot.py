from controller import NMPCController
import numpy as np
import mujoco
import mujoco.viewer
from filter import UnscentedKalmanFilter
import matplotlib.pyplot as plt

def main():
    # 1. Setup Environment
    model = mujoco.MjModel.from_xml_path('3DoFarm.xml')
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    target_body_id = model.body('target').id
    obs_mocap_id = model.body('obstacle').mocapid[0]

    # Initialize modules (NO RRT* PLANNER)
    controller = NMPCController(dt=0.02, horizon=20)
    ukf = UnscentedKalmanFilter(dt=0.02) 

    tolerance = 0.03  
    target_pos = data.xpos[target_body_id] + np.array([0.0, 0.0, 0.07])
    
    # Create a static trajectory matrix to feed the upgraded controller
    static_target_trajectory = np.tile(target_pos, (controller.N + 1, 1))
    
    swing_speed = 8.0  
    swing_distance = 0.25 
    u_prev = np.zeros(4)

    init_q = data.qpos[:4].copy()
    start_ee_pos = np.array(controller.kin.forward_kinematics_sym(init_q)).flatten()

    print("🤖 Running NMPC-Only Architecture (No Global Planner)...")

    log_ee_pos = []
    log_obs_pos = []
    
    step_count = 0
    max_steps = 400  # Timeout limit
    reached_target = False  # NEW: Flag to track actual outcome

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            raw_sensor_bus = np.array(data.sensordata)
            noisy_pos = raw_sensor_bus[0:4] + np.random.normal(0, 0.005, 4) 
            noisy_vel = raw_sensor_bus[4:8] + np.random.normal(0, 0.02, 4)
            z_meas = np.concatenate((noisy_pos, noisy_vel))

            ukf.predict(u_prev)
            q_est, dq_est = ukf.update(z_meas)

            sim_time = data.time
            data.mocap_pos[obs_mocap_id][1] = np.sin(sim_time * swing_speed) * swing_distance
            obs_pos = data.mocap_pos[obs_mocap_id].copy()

            try:
                optimal_acc = controller.solve(q_est, dq_est, static_target_trajectory, obs_pos)
            except Exception as e:
                print(f"Solver failed or trapped: {e}")
                optimal_acc = np.zeros(4)

            ee_pos_est = np.array(controller.kin.forward_kinematics_sym(q_est)).flatten()
            distance_to_final = np.linalg.norm(ee_pos_est - target_pos)

            log_ee_pos.append(ee_pos_est.copy())
            log_obs_pos.append(obs_pos.copy())

            # --- DYNAMIC OUTCOME LOGIC ---
            if distance_to_final < tolerance:
                print(f"🎯 Goal Reached (Erratic)! Terminal Error: {distance_to_final*1000:.1f} mm.")
                reached_target = True
                break
            
            if step_count > max_steps:
                print("🚨 Timeout Reached! The NMPC stalled in a local minimum.")
                break

            step_count += 1

            u_prev = optimal_acc.copy()
            data.qacc[:4] = optimal_acc
            mujoco.mj_inverse(model, data)
            data.ctrl[:4] = data.qfrc_inverse[:4].copy()
            
            for _ in range(10):
                mujoco.mj_step(model, data)
            viewer.sync()

    # --- Generate the Dynamic Plot ---
    print("📊 Generating 3D Trajectory Plot...")
    log_ee = np.array(log_ee_pos)
    log_obs = np.array(log_obs_pos)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(log_ee[:, 0], log_ee[:, 1], log_ee[:, 2], 
            color='#e67e22', linestyle='-', linewidth=3, label='NMPC-Only Trajectory')
    
    # --- NEW: Dynamic Marker Formatting ---
    if reached_target:
        marker_label = 'Erratic Convergence (Thrashing)'
        marker_color = '#f39c12'  # Amber
    else:
        marker_label = 'Local Minimum Stall (Trapped)'
        marker_color = '#c0392b'  # Dark Red
        
    ax.scatter(log_ee[-1, 0], log_ee[-1, 1], log_ee[-1, 2], 
               color=marker_color, marker='X', s=150, zorder=5, label=marker_label)
    
    ax.plot(log_obs[:, 0], log_obs[:, 1], log_obs[:, 2], 
            color='#e74c3c', linestyle='-', linewidth=2, alpha=0.7, label='Dynamic Obstacle Path')
    
    ax.scatter(start_ee_pos[0], start_ee_pos[1], start_ee_pos[2], 
               color='#3498db', s=100, edgecolors='k', label='Start Position')
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
               color='#f1c40f', s=200, marker='*', edgecolors='k', label='Target Coordinate')
    
    ax.set_title("Flat Architecture Failure Analysis", fontsize=14, pad=20)
    ax.set_xlabel("X-Axis [m]", fontsize=10, labelpad=10)
    ax.set_ylabel("Y-Axis [m]", fontsize=10, labelpad=10)
    ax.set_zlabel("Z-Axis [m]", fontsize=10, labelpad=10)
    
    ax.view_init(elev=25, azim=-45)
    ax.legend(frameon=True, facecolor='white', framealpha=1.0, 
              loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('nmpc_only_trajectory.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()