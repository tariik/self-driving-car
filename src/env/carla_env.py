from __future__ import print_function
import os
import random
import cv2
import gymnasium as gym
import numpy as np
from PIL import Image
import carla

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
        self.core.setup_experiment(self.experiment.config)
        
        # Initialize render settings
        self.render_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'render_output')
        os.makedirs(self.render_dir, exist_ok=True)
        self.render_counter = 0
        print(f"📷 Render frames will be saved to: {self.render_dir}")

        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()

        # Tick once and get the observations
        sensor_data = self.core.tick(None) 
        
        # Create mock sensor data with realistic dimensions
        # sensor_data = self.mock_sensor_data()
      
        observation, _ = self.experiment.get_observation(sensor_data)
        
        # Store observation for rendering
        self.last_observation = observation
        
        # Update spectator to follow hero
        self.update_spectator()

        return observation, {}

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""
        control = self.experiment.compute_action(action)
        sensor_data = self.core.tick(control)
        # sensor_data = self.mock_sensor_data()
        # TODO: transform image using semantic_segmentation.
        # TODO: using image augemtation??.
        observation, info = self.experiment.get_observation(sensor_data)
        done = self.experiment.get_done_status(observation, self.core)
        # done = False
        reward = self.experiment.compute_reward(observation, self.core)
        # reward = random.uniform(-1, 1)  # Mock reward for testing purposes
        
        # Store observation for rendering
        self.last_observation = observation
        
        # Update spectator to follow hero
        self.update_spectator()
        
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

    def render(self, mode='human'):
        """Save the current observation as an image file"""
        # Get the last observation from the experiment
        if not hasattr(self, 'last_observation') or self.last_observation is None:
            return
        
        # Take the last frame from the stack (most recent)
        observation = self.last_observation
        if len(observation.shape) == 3:
            current_frame = observation[:, :, -1]  # Shape: (84, 84)
        else:
            current_frame = observation
        
        # Convert to uint8 if needed
        if current_frame.dtype != np.uint8:
            current_frame = (current_frame * 255).astype(np.uint8)
        
        # Save the image using PIL (higher quality than cv2)
        img = Image.fromarray(current_frame, mode='L')  # 'L' mode for grayscale
        # Scale up 4x for better visibility
        img = img.resize((336, 336), Image.NEAREST)
        
        filename = os.path.join(self.render_dir, f'frame_{self.render_counter:04d}.png')
        img.save(filename)
        self.render_counter += 1
        
        if self.render_counter % 10 == 0:
            print(f"💾 Saved {self.render_counter} frames")
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'render_counter') and self.render_counter > 0:
            print(f"\n✅ Total frames saved: {self.render_counter}")
            print(f"📁 Location: {self.render_dir}")
        pass
    
    def update_spectator(self):
        """Move spectator camera to follow the hero vehicle from above"""
        if self.hero is None:
            return
        
        try:
            # Get hero transform
            hero_transform = self.hero.get_transform()
            
            # Set spectator position: above and behind the hero
            spectator_transform = carla.Transform(
                carla.Location(
                    x=hero_transform.location.x,
                    y=hero_transform.location.y,
                    z=hero_transform.location.z + 50  # 50 meters above
                ),
                carla.Rotation(
                    pitch=-90,  # Look straight down
                    yaw=hero_transform.rotation.yaw,
                    roll=0
                )
            )
            
            # Update spectator
            spectator = self.core.world.get_spectator()
            spectator.set_transform(spectator_transform)
        except Exception as e:
            # Silently ignore errors (spectator might not be available)
            pass




