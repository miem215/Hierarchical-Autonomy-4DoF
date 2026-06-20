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
        
    def solve(self, q_curr, dq_curr, target_pos):
        # Reset and clear previous constraints
        self.opti = ca.Opti() 
        X = self.opti.variable(6, self.N + 1)
        U = self.opti.variable(3, self.N)
        
        cost = 0
        for k in range(self.N):
            # Cost: Tracking error + control effort
            ee_pos = self.kin.forward_kinematics_sym(X[:3, k])
            cost += ca.sumsqr(ee_pos - target_pos) * 10
            cost += ca.sumsqr(U[:, k]) * 0.1
            
        self.opti.minimize(cost)
        
        # Constraints
        self.opti.subject_to(X[:, 0] == ca.vertcat(q_curr, dq_curr))
        for k in range(self.N):
            x_next = self.f_dynamics(X[:3, k], X[3:, k], U[:, k])
            self.opti.subject_to(X[:, k+1] == ca.vertcat(x_next[0], x_next[1]))
            
        # Solver setup
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)
        
        sol = self.opti.solve()
        return sol.value(U[:, 0]) # Return first optimal acceleration