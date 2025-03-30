
from __future__ import print_function
import os
import random
import cv2
import gymnasium as gym
import numpy as np

from src.env.base_env import BaseEnv
from src.env.carla_core import CarlaCore


class CarlaEnv(gym.Env):
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """

    def __init__(self, experiment:BaseEnv, config):
        """Initializes the environment"""
        self.experiment = experiment
        self.config = config
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()

        self.core = CarlaCore(self.config['carla'])
        # TODO: self.core.setup_experiment(self.experiment.config)

        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        # TODO: self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()

        # Tick once and get the observations
        # TODO: sensor_data = self.core.tick(None) 
        
        # Create mock sensor data with realistic dimensions
        sensor_data = self.mock_sensor_data()
      
        observation, _ = self.experiment.get_observation(sensor_data)
        
        

        return observation, {}

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""
        control = self.experiment.compute_action(action)
        # TODO: sensor_data = self.core.tick(control)
        sensor_data = self.mock_sensor_data()
        # TODO: transform image using semantic_segmentation.
        # TODO: using image augemtation??.
        observation, info = self.experiment.get_observation(sensor_data)
        # TODO: done = self.experiment.get_done_status(observation, self.core)
        done = False
         # TODO: reward = self.experiment.compute_reward(observation, self.core)
        reward = random.uniform(-1, 1)  # Mock reward for testing purposes
        return observation, reward, done, {} , info
    
    def mock_sensor_data(self):
        # Use dimensions from config or observation space
        image_height = self.observation_space.shape[0]
        image_width = self.observation_space.shape[1]
        
        # Generate random images with correct dimensions
        rgb_image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
        depth_image = np.random.rand(image_height, image_width).astype(np.float32)
        lidar_points = np.random.rand(1000, 4).astype(np.float32)
        birdview_image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)

        # Generate random sensor events
        collision_event = {
            "frame": np.random.randint(0, 1000),
            "intensity": np.random.uniform(0, 20),
            "actor_id": np.random.randint(0, 100)
        }
        lane_invasion_event = {
            "frame": np.random.randint(0, 1000),
            "crossed_lane_markings": [random.choice(["SOLID", "BROKEN"])]
        }
        gnss_event = {
            "latitude": np.random.uniform(-90, 90),
            "longitude": np.random.uniform(-180, 180),
            "altitude": np.random.uniform(0, 100)
        }
        imu_event = {
            "accelerometer": np.random.uniform(-10, 10, 3).tolist(),
            "gyroscope": np.random.uniform(-1, 1, 3).tolist()
        }
        
        image_path = os.path.join( 'data', 'rgb', '000.png')
        if os.path.exists(image_path):
            # Read image - cv2.imread loads as BGR, so convert to RGB
            rgb_image = cv2.imread(image_path)
            
            if rgb_image is not None:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                # Resize image to match the observation space dimensions
                rgb_image = cv2.resize(rgb_image, (image_width, image_height))
                
            else:
                print(f"Warning: Could not load image {image_path}. Using random image instead.")
                rgb_image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
                print(f"Warning: Image {rgb_image} is None. Using random image instead.")
        else:
            print(f"Warning: Image file {image_path} not found. Using random image instead.")
            rgb_image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
    
       
        # Full sensor data dictionary with random data for each sensor type
        sensor_data = {
            "rgb_camera": [rgb_image],
             # "depth_camera": [depth_image],
             # "lidar": [lidar_points],
            "collision": [collision_event],
            # "lane_invasion": [lane_invasion_event],
            # "gnss": [gnss_event],
            # "imu": [imu_event],
            # "birdview": [None, birdview_image]
        }
        
        return sensor_data
            
        
    
    
