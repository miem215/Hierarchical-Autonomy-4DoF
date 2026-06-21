import casadi as ca
import numpy as np
from Kinematic import KinematicsEngine

class NMPCController:
    def __init__(self, dt=0.05, horizon=20):
        self.dt = dt
        self.N = horizon
        self.kin = KinematicsEngine()
        self.opti = ca.Opti()
        
        # Define symbolic variables
        self.q = ca.SX.sym('q', 3)
        self.dq = ca.SX.sym('dq', 3)
        self.u = ca.SX.sym('u', 3)
        
        # Dynamics: Euler integration
        self.f_dynamics = ca.Function('f', [self.q, self.dq, self.u], 
                                      [self.q + self.dq * self.dt, 
                                       self.dq + self.u * self.dt])
        
        # Initialize the optimization variables
        self.X = self.opti.variable(6, self.N + 1)
        self.U = self.opti.variable(3, self.N)
        
    def solve(self, q_curr, dq_curr, target_pos, obs_pos):
        # Reset and clear previous constraints
        self.opti = ca.Opti() 
        X = self.opti.variable(6, self.N + 1)
        U = self.opti.variable(3, self.N)
        
        safe_radius_sq = 0.05**2
        cost = 0
        slack = self.opti.variable(self.N)
        W_obs = 1000  # Massive penalty for clipping the obstacle
        
        # --- 1. THE RUNNING COST & CONSTRAINTS (The Journey) ---
        for k in range(self.N):
            ee_pos = self.kin.forward_kinematics_sym(X[:3, k])
            
            # Massively reduced position weight: allow the arm to swing wide!
            cost += ca.sumsqr(ee_pos - target_pos) * 10.0
            cost += ca.sumsqr(U[:, k]) * 0.1

            cost += W_obs * slack[k]
            cost += ca.sumsqr(X[3:, k]) * 0.25

            # Hard obstacle constraint (No slack variable)
            ee_dist_sq = (ee_pos[0] - obs_pos[0])**2 + (ee_pos[1] - obs_pos[1])**2
            self.opti.subject_to(ee_dist_sq + slack[k] >= safe_radius_sq)
            self.opti.subject_to(slack[k] >= 0)
            
            # Dynamics
            x_next = self.f_dynamics(X[:3, k], X[3:, k], U[:, k])
            self.opti.subject_to(X[:, k+1] == ca.vertcat(x_next[0], x_next[1]))
            
        # --- 2. THE TERMINAL COST (The Strategic Goal) ---
        # Calculate exactly where the arm is at the final step of the horizon
        ee_final_pos = self.kin.forward_kinematics_sym(X[:3, self.N])
        
        # Add a massive reward ONLY for reaching the target at the very end
        cost += ca.sumsqr(ee_final_pos - target_pos) * 1000.0 
            
        self.opti.minimize(cost)
        
        # Initial condition constraint
        self.opti.subject_to(X[:, 0] == ca.vertcat(q_curr, dq_curr))
            
        # Solver setup
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)
        
        sol = self.opti.solve()
        return sol.value(U[:, 0]) # Return first optimal acceleration