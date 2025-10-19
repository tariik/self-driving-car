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
    
    def render_hud(self, step, reward, total_reward, done=False):
        """Render HUD with training information"""
        self.step_count = step
        self.last_reward = reward
        self.total_reward = total_reward
        
        # Semi-transparent background
        hud_surface = pygame.Surface((280, 200))
        hud_surface.set_alpha(200)
        hud_surface.fill((0, 0, 0))
        self.display.blit(hud_surface, (10, 10))
        
        # Training info
        y_offset = 20
        
        # Title
        title = self.font.render("CARLA DRL Training", True, (255, 255, 0))
        self.display.blit(title, (20, y_offset))
        y_offset += 30
        
        # Stats
        texts = [
            f"Step: {step}",
            f"FPS: {self.fps:.1f}",
            f"Reward: {reward:.4f}",
            f"Total: {total_reward:.2f}",
        ]
        
        for text in texts:
            color = (0, 255, 0)
            surface = self.small_font.render(text, True, color)
            self.display.blit(surface, (20, y_offset))
            y_offset += 25
        
        # Status
        status_text = 'DONE' if done else 'Running'
        status_color = (255, 255, 0) if done else (0, 255, 0)
        surface = self.small_font.render(f"Status: {status_text}", True, status_color)
        self.display.blit(surface, (20, y_offset))
    
    def update(self, step=0, reward=0.0, total_reward=0.0, done=False):
        """Update display with new data"""
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
        self.render_hud(step, reward, total_reward, done)
        
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

