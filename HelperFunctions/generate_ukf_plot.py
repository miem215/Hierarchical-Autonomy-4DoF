import numpy as np
import mujoco
import matplotlib.pyplot as plt
from filter import UnscentedKalmanFilter
from controller import NMPCController

def generate_ukf_plot():
    print("Running simulation to collect UKF data...")
    # Setup MuJoCo (Headless - no viewer needed)
    model = mujoco.MjModel.from_xml_path('3DoFarm.xml')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Initialize NMPC and UKF with dt=0.02 (50Hz)
    controller = NMPCController(dt=0.02, horizon=20)
    ukf = UnscentedKalmanFilter(dt=0.02) 

    target_body_id = model.body('target').id
    obs_mocap_id = model.body('obstacle').mocapid[0]
    target_pos = data.xpos[target_body_id] + np.array([0.0, 0.0, 0.07])
    
    u_prev = np.zeros(4)

    # Data collection arrays
    time_data = []
    true_vel_data = []
    noisy_vel_data = []
    est_vel_data = []

    # Run for 150 steps (3.0 seconds total at 50Hz)
    for _ in range(150):
        # 1. Ground Truth
        raw_sensor_bus = np.array(data.sensordata)
        true_vel = raw_sensor_bus[4] # Track Joint 1 Velocity
        
        # 2. Noisy Measurement
        noisy_pos = raw_sensor_bus[0:4] + np.random.normal(0, 0.005, 4) 
        # Add significant noise to velocity to show off the UKF
        noisy_vel = raw_sensor_bus[4:8] + np.random.normal(0, 0.05, 4) 
        z_meas = np.concatenate((noisy_pos, noisy_vel))

        # 3. UKF Estimate
        ukf.predict(u_prev)
        q_est, dq_est = ukf.update(z_meas)
        est_vel = dq_est[0] # Estimated Joint 1 Velocity

        # Save data
        time_data.append(data.time)
        true_vel_data.append(true_vel)
        noisy_vel_data.append(noisy_vel[0])
        est_vel_data.append(est_vel)

        # Control
        ee_pos_est = controller.kin.forward_kinematics_sym(q_est)
        data.mocap_pos[obs_mocap_id][1] = np.sin(data.time * 4.0) * 0.5
        obs_pos = data.mocap_pos[obs_mocap_id].copy()

        try:
            optimal_acc = controller.solve(q_est, dq_est, target_pos, obs_pos)
        except:
            optimal_acc = np.zeros(4)

        u_prev = optimal_acc.copy()
        data.qacc[:4] = optimal_acc
        mujoco.mj_inverse(model, data)
        data.ctrl[:4] = data.qfrc_inverse[:4].copy()
        
        # CRITICAL FIX: Step the physics 10 times to match dt=0.02
        # (10 * 0.002s = 0.02s)
        for _ in range(10):
            mujoco.mj_step(model, data)

    print("Data collected! Generating academic plot...")

    # --- Plotting the Data ---
    plt.figure(figsize=(10, 5))
    
    # Plot Noisy Data (Faint dots/lines so it doesn't overwhelm)
    plt.plot(time_data, noisy_vel_data, color='red', alpha=0.3, label='Noisy Sensor Measurement', linestyle='--')
    
    # Plot Ground Truth
    plt.plot(time_data, true_vel_data, color='black', linewidth=2, label='Ground Truth (MuJoCo)')
    
    # Plot UKF Estimate
    plt.plot(time_data, est_vel_data, color='blue', linewidth=2, label='UKF Estimate')

    plt.title('UKF Performance: Joint 1 Velocity Tracking', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Velocity (rad/s)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ukf_performance.png', dpi=300)
    print("Plot saved as 'ukf_performance.png'!")

if __name__ == '__main__':
    generate_ukf_plot()