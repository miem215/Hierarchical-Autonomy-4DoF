import casadi as ca
import numpy as np
from Kinematic import KinematicsEngine

class NMPCController:
    def __init__(self, dt=0.05, horizon=20):
        self.dt = dt
        self.N = horizon
        self.kin = KinematicsEngine()
        self.opti = ca.Opti()
        
        self.X = self.opti.variable(8, self.N + 1)
        self.U = self.opti.variable(4, self.N)
        
    def solve(self, q_curr, dq_curr, target_trajectory, obs_pos):
        """
        target_trajectory must be a numpy matrix / list of shape (N+1, 3) 
        representing the pre-computed high-level Cartesian waypoints.
        """
        self.opti = ca.Opti() 
        X = self.opti.variable(8, self.N + 1)
        U = self.opti.variable(4, self.N)
        
        safe_radius_sq = 0.3**2
        cost = 0
        slack = self.opti.variable(self.N)

        self.opti.set_initial(X, ca.repmat(ca.vertcat(q_curr, dq_curr), 1, self.N + 1))
        self.opti.set_initial(slack, 0.1) 
        
        W_obs = 100000.0  
        q_home = ca.vertcat(0.0, 0.0, 0.0, 0.0)

        # --- 1. THE RUNNING COST & CONSTRAINTS (The Journey) ---
        for k in range(self.N):
            ee_pos = self.kin.forward_kinematics_sym(X[:4, k])
            
            # UPGRADE POINT: Track the time-indexed path from RRT* instead of static goal
            ref_pos_k = target_trajectory[k, :]
            cost += ca.sumsqr(ee_pos - ref_pos_k) * 500.0
            
            cost += ca.sumsqr(U[:, k]) * 0.2
            cost += W_obs * slack[k]
            cost += ca.sumsqr(X[4:, k]) * 0.2

            # ==========================================
            # WHOLE-BODY COLLISION AVOIDANCE (3D Forces)
            # ==========================================
            q1, q2, q3 = X[0, k], X[1, k], X[2, k]
            r_elbow = 1.0 * ca.sin(q2)
            r_wrist = r_elbow + 1.0 * ca.sin(q2 + q3)
            
            elbow_x = ca.cos(q1) * r_elbow
            elbow_y = ca.sin(q1) * r_elbow
            elbow_z = 1.0 * ca.cos(q2)
            
            wrist_x = ca.cos(q1) * r_wrist
            wrist_y = ca.sin(q1) * r_wrist
            wrist_z = elbow_z + 1.0 * ca.cos(q2 + q3)
            
            mid_link3_x = (elbow_x + wrist_x) / 2.0
            mid_link3_y = (elbow_y + wrist_y) / 2.0
            mid_link3_z = (elbow_z + wrist_z) / 2.0
            
            mid_link4_x = (wrist_x + ee_pos[0]) / 2.0
            mid_link4_y = (wrist_y + ee_pos[1]) / 2.0
            mid_link4_z = (wrist_z + ee_pos[2]) / 2.0
            
            ee_dist_sq = (ee_pos[0] - obs_pos[0])**2 + (ee_pos[1] - obs_pos[1])**2 + (ee_pos[2] - obs_pos[2])**2
            self.opti.subject_to(ee_dist_sq + slack[k] >= safe_radius_sq)
            
            elbow_dist_sq = (elbow_x - obs_pos[0])**2 + (elbow_y - obs_pos[1])**2 + (elbow_z - obs_pos[2])**2
            self.opti.subject_to(elbow_dist_sq + slack[k] >= safe_radius_sq)
            
            wrist_dist_sq = (wrist_x - obs_pos[0])**2 + (wrist_y - obs_pos[1])**2 + (wrist_z - obs_pos[2])**2
            self.opti.subject_to(wrist_dist_sq + slack[k] >= safe_radius_sq)
            
            mid_link3_dist_sq = (mid_link3_x - obs_pos[0])**2 + (mid_link3_y - obs_pos[1])**2 + (mid_link3_z - obs_pos[2])**2
            self.opti.subject_to(mid_link3_dist_sq + slack[k] >= safe_radius_sq)
            
            mid_link4_dist_sq = (mid_link4_x - obs_pos[0])**2 + (mid_link4_y - obs_pos[1])**2 + (mid_link4_z - obs_pos[2])**2
            self.opti.subject_to(mid_link4_dist_sq + slack[k] >= safe_radius_sq)
            
            self.opti.subject_to(slack[k] >= 0)
            
            # Postural Objective
            posture_error = X[:4, k] - q_home
            cost += ca.mtimes([posture_error.T, ca.diag([0.01, 0.1, 0.1, 0.1]), posture_error])
            
            # Dynamics Constraints
            q_next = X[:4, k] + X[4:, k] * self.dt
            dq_next = X[4:, k] + U[:, k] * self.dt
            self.opti.subject_to(X[:, k+1] == ca.vertcat(q_next, dq_next))
            self.opti.subject_to(self.opti.bounded(-3.14, X[:4, k], 3.14))
            
        # --- 2. THE TERMINAL COST ---
        ee_final_pos = self.kin.forward_kinematics_sym(X[:4, self.N])
        ref_final_pos = target_trajectory[self.N, :]
        cost += ca.sumsqr(ee_final_pos - ref_final_pos) * 10000.0 
            
        self.opti.minimize(cost)
        self.opti.subject_to(X[:, 0] == ca.vertcat(q_curr, dq_curr))
            
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)
        
        try:
            sol = self.opti.solve()
            return sol.value(U[:, 0])
        except Exception as e:
            self.opti.debug.show_infeasibilities(1e-5)
            raise e