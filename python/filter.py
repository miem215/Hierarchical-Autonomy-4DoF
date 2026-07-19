import numpy as np
import scipy.linalg

class UnscentedKalmanFilter:
    def __init__(self, dt=0.02):
        """
        Unscented Kalman Filter for 3-DOF Manipulator State Estimation.
        State vector x: [q1, q2, q3, dq1, dq2, dq3] (6x1)
        Measurement vector z: [enc_pos1...3, enc_vel1...3] (6x1)
        """
        self.dt = dt
        self.n_x = 8  # State dimension
        self.n_z = 8  # Measurement dimension
        
        # --- UKF Tuning Parameters (Van der Merwe Scaled Sigma Points) ---
        self.alpha = 1e-3  
        self.beta = 2.0    
        self.kappa = 0.0   
        self.lambda_ = self.alpha**2 * (self.n_x + self.kappa) - self.n_x
        
        # --- Initialization ---
        self.x = np.zeros(self.n_x)                     
        self.P = np.eye(self.n_x) * 0.1                 
        
        # --- Noise Covariance Matrices ---
        # Q: Process Noise (Doubt in our mathematical dynamics)
        self.Q = np.eye(self.n_x) * 1e-4  
        
        # R: Measurement Noise (Doubt in the MuJoCo sensors)
        self.R = np.diag([0.005**2, 0.005**2, 0.005**2,  0.005**2,
                          0.02**2,  0.02**2,  0.02**2, 0.02**2])   
        
        # Calculate Unscented Transform Weights
        self.Wc, self.Wm = self._compute_weights()

    def _compute_weights(self):
        """
        Computes the weight vectors for the mean (Wm) and covariance (Wc).
        """
        Wm = np.zeros(2 * self.n_x + 1)
        Wc = np.zeros(2 * self.n_x + 1)

        # 0th sigma point weights
        Wm[0] = self.lambda_ / (self.n_x + self.lambda_)
        Wc[0] = (self.lambda_ / (self.n_x + self.lambda_)) + (1 - self.alpha**2 + self.beta)

        # 1st to 2n sigma points weights
        weight = 1.0 / (2 * (self.n_x + self.lambda_))
        Wm[1:] = weight
        Wc[1:] = weight

        return Wc, Wm

    def _generate_sigma_points(self, x, P):
        """
        Generates the 13 deterministic sigma points around the current mean.
        Uses the Cholesky decomposition of the covariance matrix.
        """
        # lower=True ensures A is lower triangular for correct mathematical extraction
        A = scipy.linalg.cholesky((self.n_x + self.lambda_) * P, lower=True)

        Xsig = np.zeros((self.n_x, 2 * self.n_x + 1))
        Xsig[:, 0] = x
        for i in range(self.n_x):
            Xsig[:, i + 1] = x + A[:, i]
            Xsig[:, i + 1 + self.n_x] = x - A[:, i]

        return Xsig 

    def _system_dynamics(self, state, u):
        """
        The nonlinear state transition function f(x, u).
        Mirrors the Euler integration perfectly.
        """
        q = state[0:4]
        dq = state[4:8]
        
        # Apply Euler integration
        q_next = q + dq * self.dt
        dq_next = dq + u * self.dt

        x_next = np.concatenate((q_next, dq_next))
        return x_next

    def predict(self, u):
        """
        Phase 1: Propagate the sigma points through the nonlinear dynamics.
        """
        # 1. Generate sigma points from current x and P
        Xsig = self._generate_sigma_points(self.x, self.P)
        
        # 2. Push each sigma point through the nonlinear dynamics
        self.Xsig_prd = np.zeros((self.n_x, 2 * self.n_x + 1))
        for i in range(2 * self.n_x + 1):
            self.Xsig_prd[:, i] = self._system_dynamics(Xsig[:, i], u)
            
        # 3. Compute predicted state mean (x_prior)
        self.x = np.zeros(self.n_x)
        for i in range(2 * self.n_x + 1):
            self.x += self.Wm[i] * self.Xsig_prd[:, i]
            
        # 4. Compute predicted covariance (P_prior)
        self.P = np.copy(self.Q) 
        for i in range(2 * self.n_x + 1):
            x_diff = self.Xsig_prd[:, i] - self.x
            self.P += self.Wc[i] * np.outer(x_diff, x_diff)

    def update(self, z):
        """
        Phase 2: Correct the predicted state using the noisy sensor measurements.
        """
        # 1. Project the predicted sigma points into the measurement space H(x)
        # Because sensors directly measure state, Zsig is exactly Xsig_prd.
        Zsig = self.Xsig_prd

        # 2. Compute predicted measurement z_prior
        z_prior = np.zeros(self.n_z)
        for i in range(2 * self.n_x + 1):
            z_prior += self.Wm[i] * Zsig[:, i]

        # 3. & 4. Compute measurement covariance S and cross-covariance T
        S = np.copy(self.R)
        T = np.zeros((self.n_x, self.n_z))

        for i in range(2 * self.n_x + 1):
            z_diff = Zsig[:, i] - z_prior
            x_diff = self.Xsig_prd[:, i] - self.x

            S += self.Wc[i] * np.outer(z_diff, z_diff)
            T += self.Wc[i] * np.outer(x_diff, z_diff)

        # 5. Calculate Kalman Gain (K)
        K = T @ np.linalg.inv(S)
    
        # 6. Update state mean and covariance
        self.x = self.x + K @ (z - z_prior)
        self.P = self.P - K @ S @ K.T
        
        return self.x[0:4], self.x[4:8]