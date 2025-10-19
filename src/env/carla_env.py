from __future__ import print_function
import os
import random
import cv2
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import carla
import math

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
        # Get project root (self-driving-car directory)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.render_dir = os.path.join(project_root, 'render_output')
        os.makedirs(self.render_dir, exist_ok=True)
        self.render_counter = 0
        print(f"📷 Render frames will be saved to: {self.render_dir}")
        
        # Spectator camera mode: 'chase' (follow behind hero) or 'route_overview'
        self.spectator_mode = 'chase'
        
        # Initialize additional sensors (Phase 1.3)
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()
        
        # Reset spectator positioning flag
        self._spectator_positioned = False
        
        # Setup additional sensors (Phase 1.3)
        self._setup_additional_sensors()
        
        # Setup random route (Phase 1.4)
        if self.experiment.use_random_routes:
            self._setup_random_route()

        # Tick once and get the observations
        sensor_data = self.core.tick(None) 
        
        # Create mock sensor data with realistic dimensions
        # sensor_data = self.mock_sensor_data()
      
        # PAPER: Pasar self.core para obtener φt y dt (Ecuación 21)
        observation, _ = self.experiment.get_observation(sensor_data, self.core)
        
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
        # PAPER: Pasar self.core para obtener φt y dt (Ecuación 21)
        observation, info = self.experiment.get_observation(sensor_data, self.core)
        done = self.experiment.get_done_status(observation, self.core)
        # done = False
        reward = self.experiment.compute_reward(observation, self.core)
        # reward = random.uniform(-1, 1)  # Mock reward for testing purposes
        
        # Store observation for rendering
        self.last_observation = observation
        
        # Update spectator to follow hero
        self.update_spectator()
        
        # Update route visualization (si hay ruta activa)
        if self.experiment.use_random_routes:
            self._update_route_visualization()
        
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
        """
        Guarda la observación exacta del agente como imagen.
        
        IMPORTANTE: Muestra EXACTAMENTE lo que el agente ve:
        - Imagen 11×11 B/W (solo los primeros 121 valores)
        - Escalada a 330×330 para visualización (11×30)
        - Sin φt ni dt en la visualización (solo imagen pura)
        """
        if not hasattr(self, 'last_observation') or self.last_observation is None:
            return
        
        observation = self.last_observation
        
        # DEBUG: Verificar dimensiones de la observación
        if self.render_counter == 0:
            print(f"\n🔍 DEBUG RENDER:")
            print(f"   Observación shape: {observation.shape}")
            print(f"   Observación total length: {len(observation)}")
        
        # PAPER: Estado = [121 píxeles imagen, φt, dt]
        # Extraer SOLO la imagen (primeros 121 valores = 11×11)
        image_flat = observation[:121]  # Primeros 121 valores
        
        if self.render_counter == 0:
            print(f"   Imagen extraída length: {len(image_flat)}")
            print(f"   Imagen min: {image_flat.min():.3f}, max: {image_flat.max():.3f}")
        
        # Reshape a 11×11
        image_2d = image_flat.reshape(11, 11)
        
        if self.render_counter == 0:
            print(f"   Imagen reshaped: {image_2d.shape}")
        
        # Desnormalizar de [-1, 1] a [0, 255]
        image_2d = ((image_2d * 128) + 128).clip(0, 255).astype(np.uint8)
        
        if self.render_counter == 0:
            print(f"   Imagen desnormalizada min: {image_2d.min()}, max: {image_2d.max()}")
            print(f"   Primeras 2 filas de 11×11:")
            for i in range(2):
                print(f"   [{i}]: {' '.join(f'{x:3d}' for x in image_2d[i])}")
        
        # Crear imagen PIL
        img = Image.fromarray(image_2d, mode='L')
        
        # Escalar a 330×330 (11×30) para mejor visualización
        img = img.resize((330, 330), Image.NEAREST)  # NEAREST mantiene píxeles cuadrados
        
        # Convertir a RGB para agregar texto
        img_rgb = img.convert('RGB')
        draw = ImageDraw.Draw(img_rgb)
        
        # Agregar información SOLO de la imagen (sin φt ni dt)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Texto simple
        text = "Agente: 11x11 pixels"
        
        # Fondo negro para el texto
        bbox = draw.textbbox((5, 5), text, font=font)
        draw.rectangle(bbox, fill='black')
        draw.text((5, 5), text, fill='lime', font=font)
        
        # Guardar
        filename = os.path.join(self.render_dir, f'frame_{self.render_counter:04d}.png')
        img_rgb.save(filename)
        self.render_counter += 1
        
        if self.render_counter % 10 == 0:
            print(f"💾 Guardados {self.render_counter} frames (imagen 11×11 pura del agente)")
    
    def close(self):
        """Clean up resources"""
        # Cleanup additional sensors
        self._cleanup_additional_sensors()
        
        if hasattr(self, 'render_counter') and self.render_counter > 0:
            print(f"\n✅ Total frames saved: {self.render_counter}")
            print(f"📁 Location: {self.render_dir}")
        pass
    
    def update_spectator(self):
        """
        Actualiza la cámara del spectator:
        - Modo 'route_overview': vista aérea fija de la ruta completa
        - Modo 'chase' (por defecto): cámara detrás del héroe siguiendo su orientación
        """
        if self.hero is None:
            return
        
        try:
            # Mantener vista fija si está en modo route_overview y ya se posicionó
            if (
                self.spectator_mode == 'route_overview'
                and self.experiment.use_random_routes
                and hasattr(self, '_spectator_positioned')
            ):
                return
            
            hero_tf = self.hero.get_transform()
            hero_loc = hero_tf.location
            hero_yaw = hero_tf.rotation.yaw
            yaw_rad = math.radians(hero_yaw)
            
            # Parámetros de chase-cam
            distance_back = 20.0   # metros detrás del coche
            height_above = 8.0     # metros por encima
            pitch_down = -12.0     # grados mirando ligeramente hacia abajo
            
            # Offset detrás del héroe según su yaw
            offset_x = -distance_back * math.cos(yaw_rad)
            offset_y = -distance_back * math.sin(yaw_rad)
            
            cam_loc = carla.Location(
                x=hero_loc.x + offset_x,
                y=hero_loc.y + offset_y,
                z=hero_loc.z + height_above
            )
            cam_rot = carla.Rotation(pitch=pitch_down, yaw=hero_yaw, roll=0.0)
            
            spectator = self.core.world.get_spectator()
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
        except Exception:
            pass

    def _setup_additional_sensors(self):
        """
        Configura sensores adicionales según el paper de Pérez-Gil et al. (2022):
        - Sensor de colisión
        - Sensor de invasión de carril
        """
        # Limpiar sensores previos si existen
        self._cleanup_additional_sensors()
        
        try:
            # Sensor de colisión
            blueprint_library = self.core.world.get_blueprint_library()
            collision_bp = blueprint_library.find('sensor.other.collision')
            collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
            self.collision_sensor = self.core.world.spawn_actor(
                collision_bp,
                collision_transform,
                attach_to=self.hero
            )
            self.collision_sensor.listen(self._on_collision)
            print("   ✅ Sensor de colisión configurado")
            
        except Exception as e:
            print(f"   ⚠️  Error configurando sensor de colisión: {e}")
            self.collision_sensor = None
        
        try:
            # Sensor de invasión de carril
            lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
            lane_invasion_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
            self.lane_invasion_sensor = self.core.world.spawn_actor(
                lane_invasion_bp,
                lane_invasion_transform,
                attach_to=self.hero
            )
            self.lane_invasion_sensor.listen(self._on_lane_invasion)
            print("   ✅ Sensor de invasión de carril configurado")
            
        except Exception as e:
            print(f"   ⚠️  Error configurando sensor de invasión: {e}")
            self.lane_invasion_sensor = None
    
    def _on_collision(self, event):
        """
        Callback para sensor de colisión
        
        Args:
            event: Evento de colisión de CARLA
        """
        self.experiment.collision_triggered = True
        
        # Información del actor con el que colisionó
        other_actor = event.other_actor
        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        print(f"   ⚠️  COLISIÓN con {other_actor.type_id} (intensidad: {intensity:.1f})")
    
    def _on_lane_invasion(self, event):
        """
        Callback para sensor de invasión de carril
        Solo penaliza invasiones de líneas sólidas (no discontinuas)
        según el paper de Pérez-Gil et al. (2022)
        
        Args:
            event: Evento de invasión de carril de CARLA
        """
        # Solo penalizar invasiones de líneas sólidas (no discontinuas)
        # Las líneas discontinuas se pueden cruzar legalmente
        for marking in event.crossed_lane_markings:
            if marking.type in [
                carla.LaneMarkingType.Solid,
                carla.LaneMarkingType.SolidSolid,
                carla.LaneMarkingType.SolidBroken,
                carla.LaneMarkingType.BrokenSolid
            ]:
                self.experiment.lane_invasion_triggered = True
                print(f"   ⚠️  INVASIÓN DE CARRIL: {marking.type}")
                break
    
    def _cleanup_additional_sensors(self):
        """Limpia los sensores adicionales"""
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except:
                pass
            self.collision_sensor = None
        
        if self.lane_invasion_sensor is not None:
            try:
                self.lane_invasion_sensor.stop()
                self.lane_invasion_sensor.destroy()
            except:
                pass
            self.lane_invasion_sensor = None
    
    def _setup_random_route(self):
        """
        Genera una ruta aleatoria usando el A* planner de CARLA (Phase 1.4)
        Como se describe en el paper de Pérez-Gil et al. (2022)
        """
        try:
            # Obtener mapa y spawn points
            world_map = self.core.world.get_map()
            spawn_points = world_map.get_spawn_points()
            
            if len(spawn_points) < 2:
                print("   ⚠️  No hay suficientes spawn points para ruta aleatoria")
                return
            
            # Seleccionar puntos aleatorios diferentes
            start_idx = np.random.randint(0, len(spawn_points))
            end_idx = np.random.randint(0, len(spawn_points))
            
            # Asegurar que start y end sean diferentes
            max_attempts = 10
            attempts = 0
            while end_idx == start_idx and attempts < max_attempts:
                end_idx = np.random.randint(0, len(spawn_points))
                attempts += 1
            
            if end_idx == start_idx:
                print("   ⚠️  No se pudo encontrar destino diferente")
                return
            
            start_transform = spawn_points[start_idx]
            end_transform = spawn_points[end_idx]
            
            # Generar ruta con GlobalRoutePlanner
            route_waypoints = self._get_route(
                start_transform.location,
                end_transform.location,
                world_map
            )
            
            if not route_waypoints:
                print("   ⚠️  No se pudo generar ruta")
                return
            
            # Guardar ruta en experiment
            self.experiment.route_waypoints = route_waypoints
            self.experiment.destination = route_waypoints[-1] if route_waypoints else None
            self.experiment.current_waypoint_index = 0
            
            # Mover hero al punto de inicio
            self.hero.set_transform(start_transform)
            
            # Calcular distancia de la ruta
            route_distance = sum(
                wp1.transform.location.distance(wp2.transform.location)
                for wp1, wp2 in zip(route_waypoints[:-1], route_waypoints[1:])
            )
            
            print(f"   🗺️  Ruta generada: {len(route_waypoints)} waypoints, {route_distance:.1f}m")
            
            # Visualizar ruta (solo en spectator, no en cámara del agente)
            self._visualize_route(route_waypoints, start_transform, end_transform)
            
        except Exception as e:
            print(f"   ⚠️  Error generando ruta aleatoria: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_route(self, start_location, end_location, world_map):
        """
        Obtener ruta entre dos puntos usando A* planner de CARLA (Phase 1.4)
        
        Args:
            start_location: Ubicación inicial
            end_location: Ubicación destino
            world_map: Mapa de CARLA
        
        Returns:
            list: Lista de waypoints de la ruta
        """
        try:
            # Importar GlobalRoutePlanner de CARLA
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            
            # Crear planner con sampling_resolution de 2.0m
            # (como se menciona en el paper)
            planner = GlobalRoutePlanner(world_map, 2.0)
            
            # Trazar ruta
            route = planner.trace_route(start_location, end_location)
            
            # Extraer solo los waypoints (route contiene tuplas de (waypoint, RoadOption))
            waypoints = [waypoint for waypoint, _ in route]
            
            return waypoints
            
        except ImportError:
            print("   ⚠️  GlobalRoutePlanner no disponible. Instala CARLA PythonAPI.")
            return []
        except Exception as e:
            print(f"   ⚠️  Error en _get_route: {e}")
            return []
    
    def _visualize_route(self, waypoints, start_transform, end_transform):
        """
        Visualiza la ruta en el mundo 3D de CARLA
        Los waypoints se dibujan SOLO en el spectator, NO aparecen en la cámara del agente
        
        Args:
            waypoints: Lista de waypoints de la ruta
            start_transform: Transform del punto inicial
            end_transform: Transform del punto final
        """
        try:
            debug = self.core.world.debug
            
            # Configuración de colores
            GREEN = carla.Color(0, 255, 0)      # Verde para waypoints normales
            BLUE = carla.Color(0, 100, 255)     # Azul para inicio
            RED = carla.Color(255, 0, 0)        # Rojo para destino
            YELLOW = carla.Color(255, 255, 0)   # Amarillo para waypoint actual
            
            # Lifetime: cuánto tiempo permanecen visibles (segundos)
            # Usar -1 para permanente, o un número para que desaparezca
            lifetime = 120.0  # 2 minutos
            
            # 1. Marcar punto de INICIO (azul, MUY GRANDE, MUY ALTO)
            debug.draw_point(
                start_transform.location + carla.Location(z=5.0),  # 5 metros arriba
                size=0.5,  # 5x más grande
                color=BLUE,
                life_time=lifetime
            )
            debug.draw_string(
                start_transform.location + carla.Location(z=7.0),  # Texto más arriba
                "START",
                color=BLUE,
                life_time=lifetime
            )
            
            # 2. Marcar punto de DESTINO (rojo, MUY GRANDE, MUY ALTO)
            debug.draw_point(
                end_transform.location + carla.Location(z=5.0),  # 5 metros arriba
                size=0.5,  # 5x más grande
                color=RED,
                life_time=lifetime
            )
            debug.draw_string(
                end_transform.location + carla.Location(z=7.0),  # Texto más arriba
                "GOAL",
                color=RED,
                life_time=lifetime
            )
            
            # 3. Dibujar todos los waypoints de la ruta (verde, MUY VISIBLES, MUY ALTO)
            for i, wp in enumerate(waypoints):
                location = wp.transform.location
                
                # Punto verde GRANDE para cada waypoint (5 metros arriba)
                debug.draw_point(
                    location + carla.Location(z=5.0),  # 5 metros arriba
                    size=0.15,  # 3x más grande
                    color=GREEN,
                    life_time=lifetime
                )
                
                # Cada 10 waypoints, dibujar número MÁS GRANDE
                if i % 10 == 0:
                    debug.draw_string(
                        location + carla.Location(z=6.5),  # Texto más arriba
                        str(i),
                        color=GREEN,
                        life_time=lifetime
                    )
            
            # 4. Dibujar líneas GRUESAS conectando los waypoints (MUY ALTO)
            for i in range(len(waypoints) - 1):
                wp1 = waypoints[i]
                wp2 = waypoints[i + 1]
                
                debug.draw_line(
                    wp1.transform.location + carla.Location(z=5.0),  # 5 metros arriba
                    wp2.transform.location + carla.Location(z=5.0),  # 5 metros arriba
                    thickness=0.15,  # 3x más grueso
                    color=GREEN,
                    life_time=lifetime
                )
            
            print(f"   👁️  Ruta visualizada: {len(waypoints)} waypoints (verde), START (azul), GOAL (rojo)")
            
            # Posicionar cámara del spectator para ver toda la ruta
            self._position_spectator_for_route(waypoints)
            
        except Exception as e:
            print(f"   ⚠️  Error visualizando ruta: {e}")
    
    def _position_spectator_for_route(self, waypoints):
        """
        Posiciona el spectator para tener una vista aérea de toda la ruta
        
        Args:
            waypoints: Lista de waypoints de la ruta
        """
        try:
            if not waypoints:
                return
            
            # Calcular centro de la ruta
            locs = [wp.transform.location for wp in waypoints]
            center_x = sum(loc.x for loc in locs) / len(locs)
            center_y = sum(loc.y for loc in locs) / len(locs)
            center_z = sum(loc.z for loc in locs) / len(locs)
            
            # Calcular dimensiones de la ruta
            min_x = min(loc.x for loc in locs)
            max_x = max(loc.x for loc in locs)
            min_y = min(loc.y for loc in locs)
            max_y = max(loc.y for loc in locs)
            
            width = max_x - min_x
            height = max_y - min_y
            max_dim = max(width, height)
            
            # Altura de la cámara MUY ALTA para vista aérea panorámica
            # Más grande la ruta, más alta la cámara
            camera_height = max(300.0, max_dim * 2.5)  # Antes: max(150, 1.5x). Ahora más alto.
            
            # Posicionar spectator sobre el centro de la ruta, mirando hacia abajo
            spectator_location = carla.Location(
                x=center_x,
                y=center_y,
                z=center_z + camera_height
            )
            
            spectator_rotation = carla.Rotation(
                pitch=-90,  # Mirando hacia abajo
                yaw=0,
                roll=0
            )
            
            spectator_transform = carla.Transform(spectator_location, spectator_rotation)
            spectator = self.core.world.get_spectator()
            spectator.set_transform(spectator_transform)
            
            # Marcar que la cámara ya fue posicionada (para no moverla en update_spectator)
            self._spectator_positioned = True
            
            # Si se desea mantener overview fijo, usar modo 'route_overview'
            # Si se prefiere persecución detrás del coche, el modo por defecto 'chase' seguirá al héroe
            print(f"   📹 Cámara spectator posicionada a {camera_height:.0f}m sobre la ruta")
            
        except Exception as e:
            print(f"   ⚠️  Error posicionando spectator: {e}")

    def _update_route_visualization(self):
        """
        Actualiza la visualización para marcar el waypoint actual
        Llama esto en cada step para ver el progreso
        """
        try:
            if not self.experiment.route_waypoints:
                return
            
            idx = self.experiment.current_waypoint_index
            if idx >= len(self.experiment.route_waypoints):
                return
            
            current_wp = self.experiment.route_waypoints[idx]
            debug = self.core.world.debug
            
            # Marcar waypoint actual en amarillo (más visible, MUY ALTO)
            debug.draw_point(
                current_wp.transform.location + carla.Location(z=5.0),  # 5 metros arriba
                size=0.2,  # Un poco más grande que los verdes
                color=carla.Color(255, 255, 0),
                life_time=0.5  # Solo 0.5 segundos (se actualiza constantemente)
            )
            
            # Marcar posición del HERO con punto grande y texto para verlo desde muy alto
            if self.hero is not None:
                hero_loc = self.hero.get_location()
                debug.draw_point(
                    hero_loc + carla.Location(z=6.0),
                    size=0.6,  # más grande para alta altitud
                    color=carla.Color(255, 0, 255),  # magenta
                    life_time=0.5
                )
                debug.draw_string(
                    hero_loc + carla.Location(z=7.0),
                    "HERO",
                    color=carla.Color(255, 0, 255),
                    life_time=0.5
                )
        except Exception:
            pass  # Silencioso para no llenar logs




