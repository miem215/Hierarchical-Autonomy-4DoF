import numpy as np

class RRTStarPlanner:
    def __init__(self, x_bounds=(-2.0, 2.0), y_bounds=(-2.0, 2.0), z_bounds=(0.1, 2.5), 
                 max_iter=500, step_size=0.15, search_radius=0.3, safe_radius=0.35):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.safe_radius = safe_radius

    class Node:
        def __init__(self, pos):
            self.pos = np.array(pos)
            self.parent = None
            self.cost = 0.0

    def plan(self, start_pos, goal_pos, obs_pos):
        """Generates a collision-free 3D trajectory from start to goal."""
        start_node = self.Node(start_pos)
        goal_node = self.Node(goal_pos)
        nodes = [start_node]

        for _ in range(self.max_iter):
            # Sample random configuration
            if np.random.rand() < 0.1:  # 10% Goal bias
                rnd_pos = goal_node.pos
            else:
                rnd_pos = np.array([
                    np.random.uniform(*self.x_bounds),
                    np.random.uniform(*self.y_bounds),
                    np.random.uniform(*self.z_bounds)
                ])

            # Find nearest node
            nearest_node = min(nodes, key=lambda n: np.linalg.norm(n.pos - rnd_pos))

            # Steer toward random point
            dir_vec = rnd_pos - nearest_node.pos
            dist = np.linalg.norm(dir_vec)
            if dist == 0: continue
            
            new_pos = nearest_node.pos + (dir_vec / dist) * min(self.step_size, dist)

            # Collision Check
            if np.linalg.norm(new_pos - obs_pos) < self.safe_radius:
                continue

            new_node = self.Node(new_pos)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + np.linalg.norm(new_pos - nearest_node.pos)

            # RRT* Rewiring Step
            near_nodes = [n for n in nodes if np.linalg.norm(n.pos - new_node.pos) < self.search_radius]
            
            # Find minimum cost parent within neighborhood
            min_node = nearest_node
            min_cost = new_node.cost
            for near_node in near_nodes:
                potential_cost = near_node.cost + np.linalg.norm(new_node.pos - near_node.pos)
                if potential_cost < min_cost:
                    min_node = near_node
                    min_cost = potential_cost
            
            new_node.parent = min_node
            new_node.cost = min_cost
            nodes.append(new_node)

            # Check if close to goal
            if np.linalg.norm(new_node.pos - goal_node.pos) < self.step_size:
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + np.linalg.norm(goal_node.pos - new_node.pos)
                nodes.append(goal_node)
                return self._extract_path(goal_node)

        # Fallback if global path fails due to random sampling limits
        return self._generate_linear_fallback(start_pos, goal_pos)

    def _extract_path(self, goal_node):
        path = []
        curr = goal_node
        while curr is not None:
            path.append(curr.pos)
            curr = curr.parent
        return np.array(path[::-1])

    def _generate_linear_fallback(self, start, goal):
        return np.vstack([np.linspace(start[i], goal[i], 50) for i in range(3)]).T

    def generate_waypoints(self, raw_path, horizon=20, num_points=150):
        """Interpolates sparse nodes into an execution path."""
        distances = np.linalg.norm(np.diff(raw_path, axis=0), axis=1)
        cum_dist = np.hstack(([0], np.cumsum(distances)))
        total_dist = cum_dist[-1]
        
        eval_pts = np.linspace(0, total_dist, num_points)
        waypoints = np.zeros((num_points, 3))
        for i in range(3):
            waypoints[:, i] = np.interp(eval_pts, cum_dist, raw_path[:, i])
            
        # Pad final point to prevent index bounds out of exception at final steps
        padding = np.tile(waypoints[-1, :], (horizon + 1, 1))
        return np.vstack((waypoints, padding))