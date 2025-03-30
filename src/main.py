import os
import sys

import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.agents.drl_flatten_agent import DRLFlattenAgent
from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv


def main():
    # 1. Set up the configuration
    config = {
        "carla": {
            "host": "localhost",
            "port": 2000,
            "timeout": 10.0,
            "timestep": 0.1,
            "retries_on_error": 5,
            "resolution_x": 84,
            "resolution_y": 84,
            "quality_level": "Epic"
        },
        "others": {
            "framestack": 4,
            "max_time_idle": 100,
            "max_time_episode": 500,
        }
    }
    
    # 2. Extend the base experiment config with custom settings
    experiment_config = BASE_EXPERIMENT_CONFIG.copy()
    experiment_config["hero"]["sensors"] = {
        "rgb_camera": {
            "type": "sensor.camera.rgb",
            "x": 0.0, "y": 0.0, "z": 100.0,
            "roll": 0.0, "pitch": -90.0, "yaw": 0.0,
            "width": 84, "height": 84, "fov": 100,
            "size": 11
        }
    }
    experiment_config["hero"]["spawn_points"] = ["134.0,195.0,2.0,0.0,0.0,0.0"]
    experiment_config["town"] = "Town05_Opt"
    experiment_config["weather"] = "ClearNoon"
    experiment_config["others"] = config["others"]
    
    # 3. Create the experiment and environment
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    
    #env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = DRLFlattenAgent(env, seed=0)

    state, _ = env.reset()
    state, _ = env.reset()
    print(f"State type: {type(state)}")
    print(f"State shape: {state.shape}")
    print(f"State size: {np.prod(state.shape)}")
    # Before agent initialization
    flat_size = int(np.prod(env.observation_space.shape))
    print(f"Flattened observation space size: {flat_size}")
    
    for j in range(5):
        action = agent.act(state)
        # env.render()
        state, reward, done, _, _ = env.step(action)
        if done:
            break
    env.close()

    

if __name__ == "__main__":
    main()
