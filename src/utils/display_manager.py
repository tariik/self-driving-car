"""
Display Manager for CARLA Training Visualization
Based on manual_control.py camera system
"""
import pygame
import numpy as np
import carla
import weakref


class DisplayManager:
    """Manages pygame window for real-time visualization with CARLA camera"""
    
    def __init__(self, world, hero_vehicle, width=800, height=600, follow_spectator=False):
        """Initialize pygame display with CARLA camera
        
        Args:
            world: carla.World instance
            hero_vehicle: The hero vehicle actor to attach camera to
            width: Window width
            height: Window height
            follow_spectator: If True, the window shows the SPECTATOR view (camera is not attached to hero)
        """
        pygame.init()
        pygame.font.init()
        
        self.world = world
        self.hero_vehicle = hero_vehicle
        self.width = width
        self.height = height
        self.follow_spectator = follow_spectator
        
        # Create pygame window
        self.display = pygame.display.set_mode(
            (width, height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.display.fill((0, 0, 0))
        pygame.display.set_caption('CARLA - DRL Training')
        pygame.display.flip()
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.small_font = pygame.font.Font(pygame.font.get_default_font(), 14)
        
        # Camera setup
        self.camera_surface = None
        self.camera_sensor = None
        
        # Training stats
        self.step_count = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.fps = 0
        
        # Driving features (from paper)
        self.velocity = 0.0  # vt (m/s)
        self.distance = 0.0  # dt (m)
        self.angle = 0.0     # φt (rad)
        self.action_throttle = 0.0
        self.action_steer = 0.0
        self.action_brake = 0.0
        
        # Setup camera sensor
        self._setup_camera()
        
        print(f"✓ Display window opened: {width}x{height}")
    
    def _setup_camera(self):
        """Setup RGB camera sensor attached to hero vehicle or following spectator"""
        try:
            bp_library = self.world.get_blueprint_library()
            camera_bp = bp_library.find('sensor.camera.rgb')
            
            # Configure camera
            camera_bp.set_attribute('image_size_x', str(self.width))
            camera_bp.set_attribute('image_size_y', str(self.height))
            camera_bp.set_attribute('fov', '110')
            
            if self.follow_spectator:
                # Spawn a FREE camera at spectator transform (not attached)
                spectator_tf = self.world.get_spectator().get_transform()
                self.camera_sensor = self.world.spawn_actor(
                    camera_bp,
                    spectator_tf,
                )
            else:
                # Camera position: behind and above the vehicle
                camera_transform = carla.Transform(
                    carla.Location(x=-5.5, z=2.8),
                    carla.Rotation(pitch=-15)
                )
                
                # Spawn camera attached to hero (spring arm)
                self.camera_sensor = self.world.spawn_actor(
                    camera_bp,
                    camera_transform,
                    attach_to=self.hero_vehicle,
                    attachment_type=carla.AttachmentType.SpringArm
                )
            
            # Setup callback
            weak_self = weakref.ref(self)
            self.camera_sensor.listen(lambda image: DisplayManager._parse_image(weak_self, image))
            
            print("   ✓ Camera sensor attached")
            
        except Exception as e:
            print(f"   ⚠️  Error setting up camera: {e}")
            self.camera_sensor = None
    
    @staticmethod
    def _parse_image(weak_self, image):
        """Parse image from CARLA camera sensor"""
        self = weak_self()
        if not self:
            return
        
        try:
            # Convert CARLA image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha
            array = array[:, :, ::-1]  # BGR to RGB
            
            # Create pygame surface
            self.camera_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            
        except Exception as e:
            print(f"Error parsing image: {e}")
    
    def process_events(self):
        """Process pygame events. Returns True if should quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return True
        return False
    
    def render_hud(self, step, reward, total_reward, done=False, epsilon=0.0):
        """Render HUD with training information
        
        Args:
            step: Current step number
            reward: Last reward received
            total_reward: Accumulated reward
            done: Whether episode is terminated
            epsilon: Current exploration rate (for DQN epsilon-greedy)
        """
        self.step_count = step
        self.last_reward = reward
        self.total_reward = total_reward
        
        # Semi-transparent background (ventana profesional pequeña)
        hud_width = 220
        hud_height = 240
        hud_x = self.width - hud_width - 15  # Esquina superior derecha
        hud_y = 15
        
        hud_surface = pygame.Surface((hud_width, hud_height))
        hud_surface.set_alpha(200)  # Semi-transparente profesional
        hud_surface.fill((20, 20, 30))  # Fondo azul oscuro profesional
        self.display.blit(hud_surface, (hud_x, hud_y))
        
        # Borde decorativo
        pygame.draw.rect(self.display, (100, 150, 255), 
                        (hud_x, hud_y, hud_width, hud_height), 2)
        
        # Training info - Formato profesional con descripciones
        y_offset = hud_y + 12
        x_offset = hud_x + 10
        line_height = 16
        
        # Title profesional
        title = self.font.render("AGENT STATUS", True, (100, 200, 255))
        self.display.blit(title, (x_offset, y_offset))
        y_offset += 25
        
        # Línea separadora
        pygame.draw.line(self.display, (100, 150, 255), 
                        (x_offset, y_offset), (hud_x + hud_width - 10, y_offset), 1)
        y_offset += 8
        
        # === VEHICLE STATE ===
        section_title = self.small_font.render("Vehicle State", True, (150, 200, 255))
        self.display.blit(section_title, (x_offset, y_offset))
        y_offset += line_height + 2
        
        # Estado con descripciones profesionales
        state_data = [
            ("Velocity:", f"{self.velocity:5.1f} m/s", (100, 255, 200)),
            ("Distance:", f"{self.distance:5.2f} m", (100, 255, 200)),
            ("Angle:", f"{np.degrees(self.angle):+5.1f}°", (100, 255, 200)),
        ]
        
        for label, value, color in state_data:
            # Label
            label_surface = self.small_font.render(label, True, (180, 180, 180))
            self.display.blit(label_surface, (x_offset, y_offset))
            # Value
            value_surface = self.small_font.render(value, True, color)
            self.display.blit(value_surface, (x_offset + 70, y_offset))
            y_offset += line_height
        
        y_offset += 5
        
        # === CONTROL ACTION ===
        section_title = self.small_font.render("Control Action", True, (255, 180, 150))
        self.display.blit(section_title, (x_offset, y_offset))
        y_offset += line_height + 2
        
        # Acciones con descripciones profesionales
        action_data = [
            ("Throttle:", f"{self.action_throttle:4.2f}", (255, 200, 150)),
            ("Steering:", f"{self.action_steer:+5.2f}", (255, 200, 150)),
            ("Brake:", f"{self.action_brake:4.2f}", (255, 200, 150)),
        ]
        
        for label, value, color in action_data:
            # Label
            label_surface = self.small_font.render(label, True, (180, 180, 180))
            self.display.blit(label_surface, (x_offset, y_offset))
            # Value
            value_surface = self.small_font.render(value, True, color)
            self.display.blit(value_surface, (x_offset + 70, y_offset))
            y_offset += line_height
        
        y_offset += 5
        
        # === TRAINING METRICS ===
        section_title = self.small_font.render("Training", True, (150, 255, 150))
        self.display.blit(section_title, (x_offset, y_offset))
        y_offset += line_height + 2
        
        # Métricas con descripciones profesionales
        training_data = [
            ("Step:", f"{step}", (150, 255, 150)),
            ("Reward:", f"{reward:+6.3f}", (150, 255, 150)),
            ("Epsilon:", f"{epsilon:.4f}", (255, 255, 100)),  # Exploración DQN
        ]
        
        for label, value, color in training_data:
            # Label
            label_surface = self.small_font.render(label, True, (180, 180, 180))
            self.display.blit(label_surface, (x_offset, y_offset))
            # Value
            value_surface = self.small_font.render(value, True, color)
            self.display.blit(value_surface, (x_offset + 70, y_offset))
            y_offset += line_height
        
        # Status badge profesional en la parte inferior
        y_offset += 3
        status_text = '● TERMINATED' if done else '● RUNNING'
        status_color = (255, 100, 100) if done else (100, 255, 100)
        status_surface = self.small_font.render(status_text, True, status_color)
        self.display.blit(status_surface, (x_offset, y_offset))
    
    def update(self, step=0, reward=0.0, total_reward=0.0, done=False, 
               velocity=0.0, distance=0.0, angle=0.0,
               throttle=0.0, steer=0.0, brake=0.0, epsilon=0.0):
        """Update display with new data
        
        Args:
            step: Current step number
            reward: Current step reward
            total_reward: Cumulative reward
            done: Episode done flag
            velocity: Vehicle velocity vt (m/s)
            distance: Distance to lane center dt (m)
            angle: Angle with respect to lane φt (rad)
            throttle: Throttle action
            steer: Steering action
            brake: Brake action
            epsilon: Exploration rate (DQN epsilon-greedy)
        """
        # Update state variables
        self.velocity = velocity
        self.distance = distance
        self.angle = angle
        self.action_throttle = throttle
        self.action_steer = steer
        self.action_brake = brake
        
        # If following spectator, sync camera to spectator transform each frame
        if self.follow_spectator and self.camera_sensor is not None:
            try:
                spectator_tf = self.world.get_spectator().get_transform()
                self.camera_sensor.set_transform(spectator_tf)
            except Exception:
                pass
        
        # Clear display
        self.display.fill((0, 0, 0))
        
        # Render camera view if available
        if self.camera_surface is not None:
            self.display.blit(self.camera_surface, (0, 0))
        
        # Render HUD
        self.render_hud(step, reward, total_reward, done, epsilon)
        
        # Update display
        pygame.display.flip()
        
        # Update FPS
        self.clock.tick(30)  # Limit to 30 FPS
        self.fps = self.clock.get_fps()
    
    def close(self):
        """Close pygame display and destroy camera"""
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        pygame.quit()
        print("   ✓ Display closed")

