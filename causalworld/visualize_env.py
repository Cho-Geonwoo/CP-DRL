import os
import numpy as np
import pybullet
import matplotlib.pyplot as plt
from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld

os.makedirs("env_visualizations", exist_ok=True)

TASK_VARIANTS = {
    "A": {
        "tool_block_mass": 0.25,
        "tool_block_size": 0.05,
        "nums_objects": 3,
        "activate_sparse_reward": False,
        "fractional_reward_weight": 1.0,
        "dense_reward_weights": [1.0, 0.0, 0.0, 0.0, 0.0],
    },
    "D": {
        "tool_block_mass": 0.55,
        "tool_block_size": 0.065,
        "nums_objects": 3,
        "activate_sparse_reward": False,
        "fractional_reward_weight": 0.5,
        "dense_reward_weights": [0.5, 0.5, 0.0, 0.0, 0.0],
    },
    "G": {
        "tool_block_mass": 0.9,
        "tool_block_size": 0.08,
        "nums_objects": 3,
        "activate_sparse_reward": True,
        "fractional_reward_weight": 0.0,
        "dense_reward_weights": [0.0, 0.0, 1.0, 0.0, 0.0],
    },
    "J": {
        "tool_block_mass": 1.5,
        "tool_block_size": 0.1,
        "nums_objects": 3,
        "activate_sparse_reward": True,
        "fractional_reward_weight": 0.0,
        "dense_reward_weights": [0.0, 0.0, 0.0, 1.0, 0.0],
    },
}


def capture_env_image(config):
    task = generate_task(
        task_generator_id="general",
        variables_space="space_a_b",
        tool_block_mass=config["tool_block_mass"],
        tool_block_size=config["tool_block_size"],
        nums_objects=config["nums_objects"],
        activate_sparse_reward=config["activate_sparse_reward"],
        fractional_reward_weight=config.get("fractional_reward_weight", 1.0),
        dense_reward_weights=np.array(
            config.get("dense_reward_weights", []), dtype=np.float64
        ),
    )

    env = CausalWorld(task=task, enable_visualization=False)
    env.reset()

    view_matrix = pybullet.computeViewMatrix(
        cameraEyePosition=[0.5, 0.5, 1.0],
        cameraTargetPosition=[0.0, 0.0, 0.0],
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    proj_matrix = pybullet.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=3.1
    )

    _, _, rgb, _, _ = pybullet.getCameraImage(
        width=480, height=480, viewMatrix=view_matrix, projectionMatrix=proj_matrix
    )

    rgb_img = np.reshape(rgb, (480, 480, 4))[:, :, :3]
    env.close()
    return rgb_img.astype(np.uint8)


images = [capture_env_image(TASK_VARIANTS[t]) for t in ["A", "D", "G", "J"]]
titles = ["T1", "T2", "T3", "T4"]

fig, axs = plt.subplots(1, 4, figsize=(8, 8))
for ax, img, title in zip(axs, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.title.set_fontsize(16)
    ax.axis("off")

plt.tight_layout()
plt.savefig("env_visualizations/tasks_combined.pdf", bbox_inches="tight", pad_inches=0)
plt.close()
