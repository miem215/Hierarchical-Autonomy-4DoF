# test_controller.py
from controller import NMPCController # assuming your class is in controller.py
import numpy as np

# 1. Setup
controller = NMPCController()
q_init = np.array([0.1, 0.1, 0.1])
dq_init = np.array([0.0, 0.0, 0.0])
target = np.array([0.5, 0.2, 0.5])

# 2. Run a single solver step
try:
    optimal_acc = controller.solve(q_init, dq_init, target)
    print(f"Controller successfully generated acceleration: {optimal_acc}")
except Exception as e:
    print(f"Solver failed: {e}")