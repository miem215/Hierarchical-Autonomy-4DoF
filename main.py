# test_controller.py
from controller import NMPCController # assuming your class is in controller.py
import numpy as np
import mujoco
import mujoco.viewer


def main():
# 1. Setup
    model = mujoco.MjModel.from_xml_path('3DoFarm.xml')
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    mujoco.mj_forward(model, data)

    ee_site_id = model.site('end_effector').id
    target_body_id = model.body('target').id
    obs_body_id = model.body('obstacle').id

    controller = NMPCController(dt=0.02, horizon=20)

    tolerance = 0.03  # 1 cm tolerance
    target_reached = False

    
    target_pos = data.xpos[target_body_id] + np.array([0.0, 0.0, 0.07])

    with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                # Step the physics
                ee_pos = data.site_xpos[ee_site_id].copy()
                q_init = data.qpos[:3].copy()
                dq_init = data.qvel[:3].copy()

                # Get actual End-Effector and target id
                obs_pos = data.xpos[obs_body_id].copy()
                distance = np.linalg.norm(ee_pos - target_pos)

                if distance < tolerance:
                    if not target_reached:
                        print(f"Target Reached! Error: {distance*1000:.1f} mm. Switching to Hold Mode.")
                        target_reached = True
                
                # HOLD MODE: Command exactly zero acceleration
                # Inverse dynamics will automatically hold the arm against gravity
                    optimal_acc = np.zeros(3)

                else:

                    try:
                        optimal_acc = controller.solve(q_init, dq_init, target_pos, obs_pos)
                        
                    except Exception as e:
                        print(f"Solver failed: {e}")
                        optimal_acc = np.zeros(3)

                # 2. Tell MuJoCo what acceleration you WANT to achieve
                data.qacc[:3] = optimal_acc

                # 3. Run Inverse Dynamics (This updates data.qfrc_inverse)
                mujoco.mj_inverse(model, data)

                # 4. Extract the calculated required torque
                required_torque = data.qfrc_inverse[:3].copy()

                # 5. Send the torque to the actual actuators
                data.ctrl[:3] = required_torque

                # 6. Step the physics forward
                mujoco.mj_step(model, data)
                viewer.sync()


if __name__ == '__main__':
    main()