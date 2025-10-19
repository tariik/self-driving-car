import math
import numpy as np
import cv2
from gymnasium.spaces import Box, Discrete
from src.env.helper import join_dicts
import carla

BASE_EXPERIMENT_CONFIG = {
    "hero": {
        "blueprint": "vehicle.lincoln.mkz_2017",
        "sensors": {  # Go to sensors/factory.py to check all the available sensors
            # "sensor_name1": {
            #     "type": blueprint,
            #     "attribute1": attribute_value1,
            #     "attribute2": attribute_value2
            # }
            # "sensor_name2": {
            #     "type": blueprint,
            #     "attribute_name1": attribute_value1,
            #     "attribute_name2": attribute_value2
            # }
        },
        "spawn_points": [
            # "0,0,0,0,0,0",  # x,y,z,roll,pitch,yaw
        ]
    },
    "background_activity": {
        "n_vehicles": 0,
        "n_walkers": 0,
        "tm_hybrid_mode": True,
        "seed": None
    },
    "town": "Town05_Opt",
    "weather": 'ClearNoon'
}


class BaseEnv:
    def __init__(self, config={}):
        self.config = join_dicts(BASE_EXPERIMENT_CONFIG, config)
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_time_episode = self.config["others"]["max_time_episode"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_heading_deviation = 0
        self.last_action = None

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""

        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_time_episode = False

        # hero variables
        self.last_location = None
        self.last_velocity = 0

        # Sensor stack
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        self.last_heading_deviation = 0

    def get_action_space(self):
        """Returns the action space, in this case, a discrete space"""
        return Discrete(len(self.get_actions()))

    def get_observation_space(self):
        # Use 1 channel for grayscale images instead of 3 for RGB
        num_of_channels = 1
        image_space = Box(
            low=0.0,
            high=255.0,
            shape=(
                self.config["hero"]["sensors"]["rgb_camera"]["size"],
                self.config["hero"]["sensors"]["rgb_camera"]["size"],
                num_of_channels * self.frame_stack,  # 1 channel * 4 frames = 4 total channels
            ),
            dtype=np.uint8,
        )
        return image_space

    def get_actions(self):
        return {
            0: [0.0, 0.00, 0.0, False, False],  # Coast
            1: [0.0, 0.00, 1.0, False, False],  # Apply Break
            2: [0.0, 0.75, 0.0, False, False],  # Right
            3: [0.0, 0.50, 0.0, False, False],  # Right
            4: [0.0, 0.25, 0.0, False, False],  # Right
            5: [0.0, -0.75, 0.0, False, False],  # Left
            6: [0.0, -0.50, 0.0, False, False],  # Left
            7: [0.0, -0.25, 0.0, False, False],  # Left
            8: [0.3, 0.00, 0.0, False, False],  # Straight
            9: [0.3, 0.75, 0.0, False, False],  # Right
            10: [0.3, 0.50, 0.0, False, False],  # Right
            11: [0.3, 0.25, 0.0, False, False],  # Right
            12: [0.3, -0.75, 0.0, False, False],  # Left
            13: [0.3, -0.50, 0.0, False, False],  # Left
            14: [0.3, -0.25, 0.0, False, False],  # Left
            15: [0.6, 0.00, 0.0, False, False],  # Straight
            16: [0.6, 0.75, 0.0, False, False],  # Right
            17: [0.6, 0.50, 0.0, False, False],  # Right
            18: [0.6, 0.25, 0.0, False, False],  # Right
            19: [0.6, -0.75, 0.0, False, False],  # Left
            20: [0.6, -0.50, 0.0, False, False],  # Left
            21: [0.6, -0.25, 0.0, False, False],  # Left
            22: [1.0, 0.00, 0.0, False, False],  # Straight
            23: [1.0, 0.75, 0.0, False, False],  # Right
            24: [1.0, 0.50, 0.0, False, False],  # Right
            25: [1.0, 0.25, 0.0, False, False],  # Right
            26: [1.0, -0.75, 0.0, False, False],  # Left
            27: [1.0, -0.50, 0.0, False, False],  # Left
            28: [1.0, -0.25, 0.0, False, False],  # Left
        }

    def compute_action(self, action):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero"""
        action_control = self.get_actions()[int(action)]
        action = carla.VehicleControl()
        action.throttle = action_control[0]
        action.steer = action_control[1]
        action.brake = action_control[2]
        action.reverse = action_control[3]
        action.hand_brake = action_control[4]

        self.last_action = action

        return action

    def get_observation(self, sensor_data):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
       
        # Convert RGB to grayscale (B/W) to reduce dimensionality
        image = self.post_process_image(sensor_data, normalized=True, grayscale=True)

        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = self.prev_image_0
            self.prev_image_2 = self.prev_image_1

        images = image

        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0, images], axis=2)
        if self.frame_stack >= 3 and images is not None:
            images = np.concatenate([self.prev_image_1, images], axis=2)
        if self.frame_stack >= 4 and images is not None:
            images = np.concatenate([self.prev_image_2, images], axis=2)

        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image
        

        return images, {}

    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        #from collections import namedtuple
        #Velocity = namedtuple("Velocity", ["x", "y", "z"])
        #vel = Velocity(30.0, 0.0, 0.0)  # Replace with hero.get_velocity() when available
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        hero = core.hero
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode
        self.done_falling = hero.get_location().z < -0.5
        return self.done_time_idle or self.done_falling or self.done_time_episode

    def compute_reward(self, observation, core):
        """Computes the reward"""
        def unit_vector(vector):
            return vector / np.linalg.norm(vector)
        def compute_angle(u, v):
            return -math.atan2(u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1])
        def find_current_waypoint(map_, hero):
            return map_.get_waypoint(hero.get_location(), project_to_road=False, lane_type=carla.LaneType.Any)
        def inside_lane(waypoint, allowed_types):
            if waypoint is not None:
                return waypoint.lane_type in allowed_types
            return False

        world = core.world
        hero = core.hero
        map_ = core.map

        # Hero-related variables
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)
        hero_heading = hero.get_transform().get_forward_vector()
        hero_heading = [hero_heading.x, hero_heading.y]

        # Initialize last location
        if self.last_location == None:
            self.last_location = hero_location

        # Compute deltas
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
                            np.square(hero_location.y - self.last_location.y)))
        delta_velocity = hero_velocity - self.last_velocity

        # Update variables
        self.last_location = hero_location
        self.last_velocity = hero_velocity

        # Reward if going forward
        reward = delta_distance

        # Reward if going faster than last step
        if hero_velocity < 20.0:
            reward += 0.05 * delta_velocity

        # La duracion de estas infracciones deberia ser 2 segundos?
        # Penalize if not inside the lane
        closest_waypoint = map_.get_waypoint(
            hero_location,
            project_to_road=False,
            lane_type=carla.LaneType.Any
        )
        if closest_waypoint is None or closest_waypoint.lane_type not in self.allowed_types:
            reward += -0.5
            self.last_heading_deviation = math.pi
        else:
            if not closest_waypoint.is_junction:
                wp_heading = closest_waypoint.transform.get_forward_vector()
                wp_heading = [wp_heading.x, wp_heading.y]
                angle = compute_angle(hero_heading, wp_heading)
                self.last_heading_deviation = abs(angle)

                if np.dot(hero_heading, wp_heading) < 0:
                    # We are going in the wrong direction
                    reward += -0.5

                else:
                    if abs(math.sin(angle)) > 0.4:
                        if self.last_action == None:
                            self.last_action = carla.VehicleControl()

                        if self.last_action.steer * math.sin(angle) >= 0:
                            reward -= 0.05
            else:
                self.last_heading_deviation = 0

        if self.done_falling:
            reward += -40
        if self.done_time_idle:
            print("Done idle")
            reward += -100
        if self.done_time_episode:
            print("Done max time")
            reward += 100

        return reward
    
    
    def post_process_image(self,sensor_data, normalized=True, grayscale=True):
        """
        Convert image to gray scale and normalize between -1 and 1 if required
        :param image:
        :param normalized:
        :param grayscale
        :return: image
        """
        # image = sensor_data['birdview'][1]
        
        
        # sensor_data format: {sensor_name: (frame, parsed_data)}
        # So sensor_data['rgb_camera'] is (frame, image)
        # We need the image which is at index [1]
        image = sensor_data['rgb_camera'][1]
      
        if isinstance(image, list):
            image = image[0]
        
        # Get the target size from config
        target_size = self.config["hero"]["sensors"]["rgb_camera"]["size"]
        
        # RGB to (B/W)
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Resize the image to target_size x target_size
            image = cv2.resize(image, (target_size, target_size))
            # Make sure grayscale has the channel dimension
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
        else:
            # Keep RGB (3 channels)
            # Resize the image to target_size x target_size
            image = cv2.resize(image, (target_size, target_size))
            # Make sure it has 3 channels
            if len(image.shape) == 2:
                # If somehow it's grayscale, add channel dimension
                image = image[:, :, np.newaxis]
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # If it has alpha channel, remove it
                image = image[:, :, :3]
        

        if normalized:
            return (image.astype(np.float32) - 128) / 128
        else:
            return image.astype(np.uint8)