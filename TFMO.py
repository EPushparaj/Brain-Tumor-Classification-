import numpy as np
import time


def TFMOA(positions, objective_function, lb, ub, max_iter):
    # TFMOA Parameters
    c1 = 1.5  # Acceleration coefficient 1
    c2 = 1.5  # Acceleration coefficient 2
    w = 0.7  # Inertia weight
    v_max = 0.1  # Maximum velocity
    [n, dim] = positions.shape

    velocities = np.random.uniform(-v_max, v_max, size=(n, dim))
    personal_best_positions = positions.copy()
    personal_best_values = np.array([objective_function(p) for p in positions])
    global_best_position = positions[np.argmin(personal_best_values)]
    global_best_value = min(personal_best_values)
    ct = time.time()

    convergence = np.zeros(max_iter)

    for iteration in range(max_iter):
        for i in range(n):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - positions[i]) + c2 * r2 * (
                    global_best_position - positions[i])
            velocities[i] = np.clip(velocities[i], -v_max, v_max)  # Velocity clamping
            positions[i] += velocities[i]

            # Boundary enforcement
            for j in range(dim):
                positions[i][j] = np.clip(positions[i][j], ub[i,j], lb[i,j])

            # Update personal best and global best
            fitness = objective_function(positions[i])
            if fitness < personal_best_values[i]:
                personal_best_values[i] = fitness
                personal_best_positions[i] = positions[i]
            if fitness < global_best_value:
                global_best_value = fitness
                global_best_position = positions[i]
        convergence[iteration] = global_best_value
    bestsol = global_best_value
    ct = time.time() - ct
    return bestsol, convergence, global_best_position, ct
