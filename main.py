from controller import NMPCController
import numpy as np
import mujoco
import mujoco.viewer
from filter import UnscentedKalmanFilter

def main():
    # 1. Setup
    model = mujoco.MjModel.from_xml_path('3DoFarm.xml')
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    target_body_id = model.body('target').id
    obs_body_id = model.body('obstacle').id
    obs_mocap_id = model.body('obstacle').mocapid[0]

    # MATCH THE TIMESTEPS
    controller = NMPCController(dt=0.02, horizon=20)
    ukf = UnscentedKalmanFilter(dt=0.02) 

    tolerance = 0.03  
    target_reached = False
    target_pos = data.xpos[target_body_id] + np.array([0.0, 0.0, 0.07])

    
    swing_speed = 8.0  # How fast it moves
    swing_distance = 0.25 #
    
    # Initialize previous control input for the UKF prediction step
    u_prev = np.zeros(4)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # --- 1. SENSOR (Hardware Reality) ---
            raw_sensor_bus = np.array(data.sensordata)
            noisy_pos = raw_sensor_bus[0:4] + np.random.normal(0, 0.005, 4) 
            noisy_vel = raw_sensor_bus[4:8] + np.random.normal(0, 0.02, 4)
            
            # Combine into the 6x1 measurement vector Z
            z_meas = np.concatenate((noisy_pos, noisy_vel))

            # --- 2. STATE ESTIMATION (The UKF Pipeline) ---
            # Phase A: Predict where we are based on the LAST known motor command
            ukf.predict(u_prev)
            
            # Phase B: Correct the prediction using the CURRENT noisy sensors
            q_est, dq_est = ukf.update(z_meas)

            # --- 3. KINEMATICS & CONTROL (Using Clean Estimates) ---
            # Borrow the kinematics engine already built into your controller
            ee_pos_est = controller.kin.forward_kinematics_sym(q_est)

            sim_time = data.time
            data.mocap_pos[obs_mocap_id][1] = np.sin(sim_time * swing_speed) * swing_distance

            obs_pos = data.mocap_pos[obs_mocap_id].copy()
            distance = np.linalg.norm(ee_pos_est - target_pos)

            try:
            #CRITICAL FIX: Feed the CasADi solver the clean estimate, NOT the noise
                optimal_acc = controller.solve(q_est, dq_est, target_pos, obs_pos)
            except Exception as e:
                print(f"Solver failed: {e}")
                optimal_acc = np.zeros(4)


            if distance < tolerance:
              
                    print(f"Target Reached! Error: {distance*1000:.1f} mm. Switching to Hold Mode.")
   

            # --- 4. SYSTEM UPDATE ---
            # Save the calculated acceleration to feed the UKF predict step on the next loop
            u_prev = optimal_acc.copy()

            data.qacc[:4] = optimal_acc
            mujoco.mj_inverse(model, data)
            data.ctrl[:4] = data.qfrc_inverse[:4].copy()
            
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == '__main__':
    main()