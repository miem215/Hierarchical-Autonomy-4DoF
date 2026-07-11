import casadi as ca
import numpy as np
from Kinematic import KinematicsEngine

class NMPCController:
    def __init__(self, dt=0.05, horizon=20):
        self.dt = dt
        self.N = horizon
        self.kin = KinematicsEngine()
        self.opti = ca.Opti()
        
        # Initialize the optimization variables
        self.X = self.opti.variable(8, self.N + 1)
        self.U = self.opti.variable(4, self.N)
        
    def solve(self, q_curr, dq_curr, target_pos, obs_pos):
        # Reset and clear previous constraints
        self.opti = ca.Opti() 
        X = self.opti.variable(8, self.N + 1)
        U = self.opti.variable(4, self.N)
        
        # FIX 1: Increase safe radius! 
        # (0.05 EE radius + 0.04 Obstacle radius + 0.21 safety buffer)
        safe_radius_sq = 0.3**2
        
        cost = 0
        slack = self.opti.variable(self.N)

          # WARM START: Provide a reasonable initial guess so the solver doesn't start at absolute zero!
        self.opti.set_initial(X, ca.repmat(ca.vertcat(q_curr, dq_curr), 1, self.N + 1))
        self.opti.set_initial(slack, 0.1) # Start with a positive slack to prevent constraint singularity
        
        # FIX 2: Massive penalty. Make the obstacle physically terrifying to the solver.
        W_obs = 100000.0  
        
        q_home = ca.vertcat(0.0, 0,0,0)

        current_ee_pos = self.kin.forward_kinematics_sym(q_curr)
        dist_to_target = ca.norm_2(current_ee_pos - target_pos)
        
        max_dist = 2.0  
        dist_ratio = ca.fmax(0.0, ca.fmin(1.0, dist_to_target / max_dist))
        
        distal_penalty = 0.05 + (0.95 * dist_ratio)
        shoulder_penalty = 0.5 - (0.49 * dist_ratio)
        W_posture = ca.diag(ca.vertcat(0.01, shoulder_penalty, distal_penalty, distal_penalty))
        
        # --- 1. THE RUNNING COST & CONSTRAINTS (The Journey) ---
        for k in range(self.N):
            ee_pos = self.kin.forward_kinematics_sym(X[:4, k])
            
            cost += ca.sumsqr(ee_pos - target_pos) * 500.0
            cost += ca.sumsqr(U[:, k]) * 0.2

            # The Slack penalty is added to the cost
            cost += W_obs * slack[k]
            cost += ca.sumsqr(X[4:, k]) * 0.2

            # ==========================================
            # WHOLE-BODY COLLISION AVOIDANCE
            # ==========================================
            # 1. Fast 3D Kinematics for the Elbow and Wrist
            q1, q2, q3 = X[0, k], X[1, k], X[2, k]
            
            # Radial distance from the base in the X-Y plane (link length = 1.0m)
            r_elbow = 1.0 * ca.sin(q2)
            r_wrist = r_elbow + 1.0 * ca.sin(q2 + q3)
            
            # Convert radial distance and yaw (q1) into global X-Y-Z coordinates
            elbow_x = ca.cos(q1) * r_elbow
            elbow_y = ca.sin(q1) * r_elbow
            elbow_z = 1.0 * ca.cos(q2) # NEW: Calculate Z-height
            
            wrist_x = ca.cos(q1) * r_wrist
            wrist_y = ca.sin(q1) * r_wrist
            wrist_z = elbow_z + 1.0 * ca.cos(q2 + q3) # NEW: Calculate Z-height
            
            # 2. The Virtual Nodes (Midpoints of Link 3 and Link 4)
            mid_link3_x = (elbow_x + wrist_x) / 2.0
            mid_link3_y = (elbow_y + wrist_y) / 2.0
            mid_link3_z = (elbow_z + wrist_z) / 2.0 # NEW
            
            mid_link4_x = (wrist_x + ee_pos[0]) / 2.0
            mid_link4_y = (wrist_y + ee_pos[1]) / 2.0
            mid_link4_z = (wrist_z + ee_pos[2]) / 2.0 # NEW
            
            # 3. Apply the 3D spherical force fields to all nodes!
            # Adding the Z-axis distance allows objects to safely pass over/under the arm
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
            
            # Require slack to be positive
            self.opti.subject_to(slack[k] >= 0)
            # ==========================================
            
            # Postural Objective
            posture_error = X[:4, k] - q_home
            cost += ca.mtimes([posture_error.T, W_posture, posture_error])
            
            # Explicit Euler Dynamics
            q_next = X[:4, k] + X[4:, k] * self.dt
            dq_next = X[4:, k] + U[:, k] * self.dt
            self.opti.subject_to(X[:, k+1] == ca.vertcat(q_next, dq_next))
            
            self.opti.subject_to(self.opti.bounded(-3.14, X[:4, k], 3.14))
            
        # --- 2. THE TERMINAL COST (The Strategic Goal) ---
        ee_final_pos = self.kin.forward_kinematics_sym(X[:4, self.N])
        cost += ca.sumsqr(ee_final_pos - target_pos) * 1000.0 
            
        self.opti.minimize(cost)
        self.opti.subject_to(X[:, 0] == ca.vertcat(q_curr, dq_curr))
            
        # Solver setup
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)
        
        try:
            sol = self.opti.solve()
            return sol.value(U[:, 0])
        except Exception as e:
            print("\n=========================================")
            print("🚨 SOLVER CRASHED! ANALYZING VIOLATIONS...")
            print("=========================================")
            # This CasADi debug function prints exactly which constraints failed
            # The '1e-5' means it will only show violations larger than 0.00001
            self.opti.debug.show_infeasibilities(1e-5)
            print("=========================================\n")
            
            # Re-raise the error so main.py can catch it and apply 0 acceleration
            raise e