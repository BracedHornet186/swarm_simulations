import numpy as np
import random
random.seed(42)

spawn_radius = 3.0
avg_position = np.array([0.0, 0.0])
for i in range(3):
    x_pose = round(random.uniform(-spawn_radius, spawn_radius), 2)
    y_pose = round(random.uniform(-spawn_radius, spawn_radius), 2)
    avg_position += np.array([x_pose, y_pose])
avg_position /= 3
print(f"Average position of bots: {avg_position}")