import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_landscape_plot():
    print("Calculating optimization landscape...")
    
    # 1. Define the Robot's Cartesian Workspace
    x = np.linspace(-0.5, 1.5, 400)
    y = np.linspace(-0.5, 1.5, 400)
    X, Y = np.meshgrid(x, y)

    # 2. Set Parameters
    target_x, target_y = 1.2, 1.2
    obs_x, obs_y = 0.5, 0.5
    r_safe = 0.3
    W_obs = 50000.0  

    # 3. Calculate Target Tracking Cost (The Convex Bowl)
    J_track = 500.0 * ((X - target_x)**2 + (Y - target_y)**2)

    # 4. Calculate Obstacle Slack Penalty (The Non-Convex Pillar)
    dist = np.sqrt((X - obs_x)**2 + (Y - obs_y)**2)
    slack = np.maximum(0, r_safe - dist)
    J_obs = W_obs * slack

    # 5. Combine for Total Cost
    Z = J_track + J_obs
    Z_clipped = np.clip(Z, 0, 4000)

    # --- Generate the 3D Academic Plot ---
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    # Increased figure width slightly to accommodate the outside legend
    fig = plt.figure(figsize=(14, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z_clipped, cmap='viridis', edgecolor='none', alpha=0.85)
    ax.contour(X, Y, Z_clipped, zdir='z', offset=-500, cmap='viridis', levels=30, alpha=0.5)

    # Annotations & Markers
    ax.scatter(target_x, target_y, 0, color='#2ecc71', s=150, edgecolors='k', label='Target Coordinate', zorder=5)
    ax.scatter(obs_x, obs_y, 4000, color='#e74c3c', s=100, label='Obstacle Penalty Spike', zorder=5)

    # --- Overlay the Gradient Descent Trajectories ---
    start_x, start_y = -0.2, 0.2
    ax.scatter(start_x, start_y, 1500, color='#3498db', s=100, edgecolors='k', label='Start Position', zorder=5)
    
    # 1. NMPC Myopic Trap (Drives into the valley and stalls)
    trap_x = np.linspace(start_x, 0.25, 50)
    trap_y = np.linspace(start_y, 0.25, 50)
    trap_z = 500.0 * ((trap_x - target_x)**2 + (trap_y - target_y)**2) + 50
    ax.plot(trap_x, trap_y, trap_z, color='#e67e22', linewidth=4, label='Myopic Gradient Descent (Trapped)')
    ax.scatter(trap_x[-1], trap_y[-1], trap_z[-1], color='#c0392b', marker='X', s=150, zorder=6)

    # 2. RRT* Global Route (Bypasses the mountain entirely)
    rrt_x = np.array([-0.2, 0.0, 0.3, 0.8, 1.2])
    rrt_y = np.array([0.2, 0.8, 1.0, 1.1, 1.2])
    rrt_z = 500.0 * ((rrt_x - target_x)**2 + (rrt_y - target_y)**2) + 50
    ax.plot(rrt_x, rrt_y, rrt_z, color='#2ecc71', linewidth=3, linestyle='--', label='RRT* Global Path Routing')

    # Labels and Titles
    ax.set_title('Non-Convex Optimization Landscape & Local Minima', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Workspace X [m]', fontsize=11, labelpad=10)
    ax.set_ylabel('Workspace Y [m]', fontsize=11, labelpad=10)
    ax.set_zlabel('Total Cost Objective (J)', fontsize=11, labelpad=10)
    ax.set_zlim(-500, 4000)

    ax.view_init(elev=35, azim=-125)
    
    # --- MOVED LEGEND OUTSIDE THE PLOT ---
    ax.legend(frameon=True, facecolor='white', framealpha=1.0, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)
    
    # tight_layout ensures the expanded bounding box for the legend is included in the saved image
    plt.tight_layout()
    plt.savefig('optimization_landscape.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    generate_landscape_plot()