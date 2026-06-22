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

    # MATCH THE TIMESTEPS
    controller = NMPCController(dt=0.02, horizon=20)
    ukf = UnscentedKalmanFilter(dt=0.02) 

    tolerance = 0.03  
    target_reached = False
    target_pos = data.xpos[target_body_id] + np.array([0.0, 0.0, 0.07])
    
    # Initialize previous control input for the UKF prediction step
    u_prev = np.zeros(3)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # --- 1. SENSOR (Hardware Reality) ---
            raw_sensor_bus = np.array(data.sensordata)
            noisy_pos = raw_sensor_bus[0:3] + np.random.normal(0, 0.005, 3) 
            noisy_vel = raw_sensor_bus[3:6] + np.random.normal(0, 0.02, 3)
            
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

            obs_pos = data.xpos[obs_body_id].copy()
            distance = np.linalg.norm(ee_pos_est - target_pos)

            if distance < tolerance:
                if not target_reached:
                    print(f"Target Reached! Error: {distance*1000:.1f} mm. Switching to Hold Mode.")
                    target_reached = True
                optimal_acc = np.zeros(3)
            else:
                try:
                    # CRITICAL FIX: Feed the CasADi solver the clean estimate, NOT the noise
                    optimal_acc = controller.solve(q_est, dq_est, target_pos, obs_pos)
                except Exception as e:
                    print(f"Solver failed: {e}")
                    optimal_acc = np.zeros(3)

            # --- 4. SYSTEM UPDATE ---
            # Save the calculated acceleration to feed the UKF predict step on the next loop
            u_prev = optimal_acc.copy()

            data.qacc[:3] = optimal_acc
            mujoco.mj_inverse(model, data)
            data.ctrl[:3] = data.qfrc_inverse[:3].copy()
            
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == '__main__':
    main()