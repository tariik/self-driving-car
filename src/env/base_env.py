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
    "town": "Town01",  # Mapa del paper original (Pérez-Gil et al. 2022)
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
        
        # Flags para sensores adicionales (como en el paper)
        self.collision_triggered = False
        self.lane_invasion_triggered = False
        
        # Route tracking (Phase 1.4)
        self.route_waypoints = []
        self.current_waypoint_index = 0
        self.destination = None
        self.use_random_routes = config.get("use_random_routes", False)

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
        
        # Resetear flags de sensores (como en el paper)
        self.collision_triggered = False
        self.lane_invasion_triggered = False
        
        # Reset route tracking (Phase 1.4)
        self.current_waypoint_index = 0

        self.last_heading_deviation = 0

    def get_action_space(self):
        """Returns the action space, in this case, a discrete space"""
        return Discrete(len(self.get_actions()))

    def get_observation_space(self):
        # PAPER DRL-Flatten-Image (Ecuación 21):
        # S = ([Pt0, Pt1, ...Pt120], φt, dt)
        # - Imagen 11x11 grayscale = 121 píxeles
        # - φt = 1 valor (ángulo al carril)
        # - dt = 1 valor (distancia al centro)
        # Total: 121 + 1 + 1 = 123 dimensiones
        
        image_size = self.config["hero"]["sensors"]["rgb_camera"]["size"]
        image_pixels = image_size * image_size * self.frame_stack  # 11*11*1 = 121
        total_dims = image_pixels + 2  # 121 + 2 driving features (φt, dt) = 123
        
        state_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dims,),  # 1D vector de 123 dimensiones
            dtype=np.float32,
        )
        return state_space

    def get_actions(self):
        """
        PAPER: Table 1 - 27 discrete actions (DQN)
        - Steering: 9 valores [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        - Throttle: 3 valores [0, 0.5, 1]
        - Total: 9 × 3 = 27 acciones
        
        Formato: [throttle, steering, brake, reverse, hand_brake]
        """
        actions = {}
        action_id = 0
        
        # Paper: throttle [0, 0.5, 1] × steering [-1, -0.75, ..., 0.75, 1]
        for throttle in [0.0, 0.5, 1.0]:
            for steering in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
                actions[action_id] = [throttle, steering, 0.0, False, False]
                action_id += 1
        
        return actions

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

    def get_observation(self, sensor_data, core=None):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}
        :param core: CarlaCore instance (optional, para obtener driving features)

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
        
        # PAPER: Agregar φt y dt al estado (Ecuación 21)
        # S = ([Pt0, Pt1, ...Pt120], φt, dt)
        # Estado total: 121 (imagen 11x11) + 1 (φt) + 1 (dt) = 123 dimensiones
        if core is not None:
            driving_features = self.get_driving_features(core)
            dt = float(driving_features[1])  # distancia al centro
            φt = float(driving_features[2])  # ángulo al carril
            
            # Aplanar imagen y concatenar con driving features
            image_flat = images.flatten()  # 11x11x1 = 121 valores
            state = np.concatenate([image_flat, [φt], [dt]])  # 121 + 1 + 1 = 123
            
            # Info para debugging
            info = {
                'driving_features': driving_features,
                'velocity': float(driving_features[0]),  # vt
                'distance_to_center': dt,  # dt
                'angle_to_lane': φt  # φt
            }
            
            return state, info
        else:
            # Sin core, retornar solo imagen (fallback)
            return images, {}


    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        #from collections import namedtuple
        #Velocity = namedtuple("Velocity", ["x", "y", "z"])
        #vel = Velocity(30.0, 0.0, 0.0)  # Replace with hero.get_velocity() when available
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self, observation, core):
        """
        Returns whether or not the experiment has to end
        
        Según paper (Pérez-Gil et al. 2022, línea 715-716):
        "If lane_invasor or collision_sensor are activated, the episode ends"
        """
        hero = core.hero
        
        # 1. VERIFICAR COLISIÓN (termina episodio según paper)
        if self.collision_triggered:
            return True
        
        # 2. VERIFICAR INVASIÓN DE CARRIL (termina episodio según paper)
        if self.lane_invasion_triggered:
            return True
        
        # 3. Verificar timeout por inactividad
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        
        # 4. Verificar timeout por duración de episodio
        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode
        
        # 5. Verificar caída del vehículo
        self.done_falling = hero.get_location().z < -0.5
        
        # 6. Verificar si alcanzó la meta
        if self._is_goal_reached(core):
            return True
        
        return self.done_time_idle or self.done_falling or self.done_time_episode

    def compute_reward(self, observation, core):
        """
        Función de recompensa del paper de Pérez-Gil et al. (2022)
        
        R = -200 si colisión o salida de carril
        R = Σ |vt·cos(φt)| - |vt·sin(φt)| - |vt|·|dt| si en carril
        R = +100 si meta alcanzada
        
        Componentes:
        - |vt·cos(φt)|: Premia velocidad hacia adelante
        - |vt·sin(φt)|: Penaliza velocidad lateral (zigzagueo)
        - |vt|·|dt|: Penaliza desviación del centro del carril
        """
        # Obtener driving features (vt, dt, φt)
        driving_features = self.get_driving_features(core)
        vt = float(driving_features[0])  # velocidad (m/s)
        dt = float(driving_features[1])  # distancia al centro (m)
        φt = float(driving_features[2])  # ángulo al carril (rad)
        
        # 1. VERIFICAR COLISIÓN (penalización máxima)
        if self.collision_triggered:
            print("⚠️  COLISIÓN detectada: R = -200")
            return -200.0
        
        # 2. VERIFICAR INVASIÓN DE CARRIL (penalización máxima)
        if self.lane_invasion_triggered:
            print("⚠️  INVASIÓN DE CARRIL detectada: R = -200")
            return -200.0
        
        # 3. VERIFICAR SI CAYÓ (como en código original)
        hero = core.hero
        if hero.get_location().z < -0.5:
            print("⚠️  VEHÍCULO CAYÓ: R = -200")
            return -200.0
        
        # 4. VERIFICAR SI ALCANZÓ LA META (recompensa máxima)
        if self._is_goal_reached(core):
            print("🎉 META ALCANZADA: R = +100")
            return 100.0
        
        # 5. VERIFICAR TIMEOUT POR INACTIVIDAD
        if self.done_time_idle:
            print("⏱️  TIMEOUT por inactividad: R = -100")
            return -100.0
        
        # 6. VERIFICAR TIMEOUT POR DURACIÓN EPISODIO
        if self.done_time_episode:
            print("⏱️  EPISODIO COMPLETADO (max time): R = +100")
            return 100.0
        
        # 7. RECOMPENSA CONTINUA (ecuación del paper)
        # Componentes:
        # - Premia velocidad hacia adelante: |vt·cos(φt)|
        # - Penaliza velocidad lateral (zigzag): |vt·sin(φt)|
        # - Penaliza desviación del centro: |vt|·|dt|
        
        reward_forward = np.abs(vt * np.cos(φt))     # Premia ir hacia adelante
        penalty_lateral = np.abs(vt * np.sin(φt))    # Penaliza movimiento lateral
        penalty_deviation = np.abs(vt) * np.abs(dt)  # Penaliza desviación del centro
        
        reward = reward_forward - penalty_lateral - penalty_deviation
        
        # Debug (opcional - comentar para producción)
        # print(f"   R={reward:.3f} | vt={vt:.2f} | dt={dt:.2f} | φt={np.degrees(φt):.1f}°")
        
        return float(reward)
    
    def _is_goal_reached(self, core):
        """
        Verificar si el vehículo alcanzó la meta (Phase 1.4)
        
        Returns:
            bool: True si alcanzó la meta, False en caso contrario
        """
        # Si no hay ruta definida, no hay meta
        if not self.route_waypoints or self.destination is None:
            return False
        
        # Obtener ubicación actual del héroe
        hero_location = core.hero.get_location()
        
        # Calcular distancia a la meta (último waypoint)
        # Waypoint tiene transform.location, no location directamente
        distance_to_goal = hero_location.distance(self.destination.transform.location)
        
        # Actualizar índice del waypoint actual
        if self.current_waypoint_index < len(self.route_waypoints):
            current_wp = self.route_waypoints[self.current_waypoint_index]
            distance_to_current_wp = hero_location.distance(current_wp.transform.location)
            
            # Si estamos cerca del waypoint actual, avanzar al siguiente
            if distance_to_current_wp < 3.0:  # 3 metros de tolerancia
                self.current_waypoint_index += 1
        
        # Meta alcanzada si estamos a menos de 5 metros del destino
        if distance_to_goal < 5.0:
            return True
        
        return False
    
    def get_route_info(self, core):
        """
        Obtener información de progreso en la ruta (Phase 1.4)
        
        Returns:
            dict: Información de la ruta {
                'has_route': bool,
                'progress': float (0-1),
                'waypoints_completed': int,
                'total_waypoints': int,
                'distance_to_goal': float,
                'distance_to_next_wp': float
            }
        """
        if not self.route_waypoints or self.destination is None:
            return {
                'has_route': False,
                'progress': 0.0,
                'waypoints_completed': 0,
                'total_waypoints': 0,
                'distance_to_goal': 0.0,
                'distance_to_next_wp': 0.0
            }
        
        hero_location = core.hero.get_location()
        # Waypoint tiene transform.location, no location directamente
        distance_to_goal = hero_location.distance(self.destination.transform.location)
        
        # Distancia al siguiente waypoint
        distance_to_next_wp = 0.0
        if self.current_waypoint_index < len(self.route_waypoints):
            next_wp = self.route_waypoints[self.current_waypoint_index]
            distance_to_next_wp = hero_location.distance(next_wp.transform.location)
        
        # Calcular progreso
        total_waypoints = len(self.route_waypoints)
        progress = self.current_waypoint_index / total_waypoints if total_waypoints > 0 else 0.0
        
        return {
            'has_route': True,
            'progress': progress,
            'waypoints_completed': self.current_waypoint_index,
            'total_waypoints': total_waypoints,
            'distance_to_goal': distance_to_goal,
            'distance_to_next_wp': distance_to_next_wp
        }
    
    
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
        
        # 🔍 DEBUG MÁXIMO: Guardar imágenes de los primeros 3 frames
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        
        if self._debug_frame_count < 3:
            import os
            debug_dir = 'debug_output_main'
            os.makedirs(debug_dir, exist_ok=True)
            
            # Guardar RAW
            raw_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{debug_dir}/frame{self._debug_frame_count}_raw_640x480.png', raw_bgr)
            
            # Análisis variación horizontal
            gray_raw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h_var = np.mean([np.std(row) for row in gray_raw])
            
            print(f"\n{'🔍 DEBUG FRAME ' + str(self._debug_frame_count):=^80}")
            print(f"📸 Imagen RAW 640x480:")
            print(f"   Shape: {image.shape}")
            print(f"   Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.1f}")
            print(f"   H-Var: {h_var:.1f} {'❌ UNIFORME!' if h_var < 20 else '✅ OK'}")
            print(f"   💾 {debug_dir}/frame{self._debug_frame_count}_raw_640x480.png")
        
        # Get the target size from config
        target_size = self.config["hero"]["sensors"]["rgb_camera"]["size"]
        
        # RGB to (B/W)
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Resize the image to target_size x target_size
            image = cv2.resize(image, (target_size, target_size))
            
            # 🔍 DEBUG: Matriz 11x11 del primer frame
            if self._debug_frame_count < 3:
                print(f"\n📐 Matriz 11×11:")
                for i, row in enumerate(image):
                    vals = ' '.join(f'{v:3d}' for v in row)
                    std = np.std(row)
                    status = ' ←❌' if std < 5 else ''
                    print(f"   {i:2d}: {vals}{status}")
                h_var_11 = np.mean([np.std(row) for row in image])
                uniform_rows = sum(1 for row in image if np.std(row) < 5)
                print(f"\n   H-Var 11×11: {h_var_11:.1f}")
                print(f"   Filas uniformes: {uniform_rows}/11")
                print(f"   {'❌ LÍNEAS HORIZONTALES!' if h_var_11 < 10 else '✅ BUENA VARIACIÓN'}")
                print("=" * 80)
                
                self._debug_frame_count += 1
            
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
    
    def get_driving_features(self, core):
        """
        Extraer características de conducción como en el paper de Pérez-Gil et al. (2022):
        
        vt: velocidad del vehículo (m/s)
        dt: distancia al centro del carril (m)
        φt: ángulo respecto al carril (radianes)
        
        Returns:
            np.array: [vt, dt, φt] como float32
        """
        hero = core.hero
        world_map = core.map
        
        # 1. Velocidad del vehículo (vt)
        velocity = hero.get_velocity()
        vt = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s
        
        # 2. Distancia al centro del carril (dt)
        vehicle_location = hero.get_location()
        
        # Obtener waypoint más cercano (centro del carril)
        waypoint = world_map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if waypoint is None:
            # Si no hay waypoint válido, usar valores por defecto
            dt = 0.0
            φt = 0.0
        else:
            # Distancia al centro del carril
            lane_center = waypoint.transform.location
            dt = np.sqrt(
                (vehicle_location.x - lane_center.x)**2 + 
                (vehicle_location.y - lane_center.y)**2
            )
            
            # 3. Ángulo respecto al carril (φt)
            vehicle_yaw = hero.get_transform().rotation.yaw
            lane_yaw = waypoint.transform.rotation.yaw
            
            # Normalizar ángulo a [-180, 180]
            angle_diff = vehicle_yaw - lane_yaw
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            
            # Convertir a radianes
            φt = np.radians(angle_diff)
        
        return np.array([vt, dt, φt], dtype=np.float32)