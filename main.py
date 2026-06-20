import mujoco
import mujoco.viewer
import numpy as np
import time
import casadi as ca

class KinematicsEngine:
    def __init__(self):
        # Link lengths as symbolic-friendly floats
        self.L1_z = 0.2
        self.L2_z = 1.0
        self.L3_z = 1.0
        self.EE_z = 1.0

    def rot_z(self, theta):
        return ca.vertcat(
            ca.horzcat(ca.cos(theta), -ca.sin(theta), 0, 0),
            ca.horzcat(ca.sin(theta),  ca.cos(theta), 0, 0),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 1)
        )

    def rot_y(self, theta):
        return ca.vertcat(
            ca.horzcat(ca.cos(theta), 0, ca.sin(theta), 0),
            ca.horzcat(0, 1, 0, 0),
            ca.horzcat(-ca.sin(theta), 0, ca.cos(theta), 0),
            ca.horzcat(0, 0, 0, 1)
        )

    def translate_z(self, z):
        return ca.vertcat(
            ca.horzcat(1, 0, 0, 0),
            ca.horzcat(0, 1, 0, 0),
            ca.horzcat(0, 0, 1, z),
            ca.horzcat(0, 0, 0, 1)
        )

    def forward_kinematics(self, q):
        theta1, theta2, theta3 = q

        # T01: Base to Link 1 (Translate Z by 0.2, Rotate Z by theta1)
        T01 = self.translate_z(self.L1_z) @ self.rot_z(theta1)
        
        # T12: Link 1 to Link 2 (Translate Z by 1.0, Rotate Y by theta2)
        T12 = self.translate_z(self.L2_z) @ self.rot_y(theta2)
        
        # T23: Link 2 to Link 3 (Translate Z by 1.0, Rotate Y by theta3)
        T23 = self.translate_z(self.L3_z) @ self.rot_y(theta3)
        
        # T3_EE: Link 3 to End-Effector (Translate Z by 1)
        T3_EE = self.translate_z(self.EE_z)

        # Chain them together
        T0_EE = T01 @ T12 @ T23 @ T3_EE
        
        # Extract the translation vector (x, y, z)
        return T0_EE[0:3, 3]

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path('3DoFarm.xml')
    data = mujoco.MjData(model)

    
    kin_engine = KinematicsEngine()
    ee_site_id = model.site('end_effector').id
    target_id = model.body('target').id

    # Launch passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Step the physics
            mujoco.mj_step(model, data)
            viewer.sync()

            # 1. Get joint angles from MuJoCo
            q = data.qpos[:3] 

            manual_ee_pos_dm = kin_engine.forward_kinematics(q)

            # Convert the 3x1 CasADi DM into a flat 1D NumPy array
            manual_ee_pos = np.array(manual_ee_pos_dm).flatten()

            # 3. Get actual End-Effector position from MuJoCo's engine
            mujoco_ee_pos = data.site_xpos[ee_site_id]
            mujoco_tg_pos = data.xpos[target_id]

            # Calculate the error to prove our math is correct
            error = np.linalg.norm(manual_ee_pos - mujoco_ee_pos)



if __name__ == '__main__':
    main()