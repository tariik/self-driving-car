"""
Video Recorder for CARLA Training - Headless Mode
Creates MP4 videos from training sessions without needing a display
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


class VideoRecorder:
    """Records training sessions as MP4 videos (headless compatible)"""
    
    def __init__(self, output_dir='render_output', fps=10):
        """Initialize video recorder"""
        self.output_dir = output_dir
        self.fps = fps
        self.frame_counter = 0
        self.frames = []
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎥 Video recorder initialized")
        print(f"📁 Output: {output_dir}")
    
    def add_frame(self, observation, step=0, reward=0.0, total_reward=0.0, done=False):
        """Add a frame to the video
        Robustly handles the 123-dim observation vector: [121 image pixels, φt, dt]
        """
        try:
            # Build an 11x11 grayscale image from the observation
            img_array = None
            phi_t = None
            d_t = None

            if isinstance(observation, np.ndarray) and observation.ndim == 1 and observation.size >= 121:
                # State vector: [121 image, φt, dt]
                img_flat = observation[:121]
                try:
                    img_array = img_flat.reshape(11, 11)
                except Exception:
                    # Fallback to square if possible
                    side = int(np.sqrt(121))
                    img_array = img_flat.reshape(side, side)
                # Extract driving features if present
                if observation.size > 121:
                    phi_t = float(observation[121]) if observation.size > 121 else None
                    d_t = float(observation[122]) if observation.size > 122 else None
            elif isinstance(observation, np.ndarray) and observation.ndim == 2:
                # Already a 2D image
                img_array = observation
            elif isinstance(observation, np.ndarray) and observation.ndim == 3:
                # Use last channel if stack
                img_array = observation[:, :, -1]
            else:
                # Unknown format, try to coerce
                obs = np.array(observation)
                if obs.ndim == 1 and obs.size >= 121:
                    img_array = obs[:121].reshape(11, 11)
                elif obs.ndim == 2:
                    img_array = obs
                elif obs.ndim == 3:
                    img_array = obs[:, :, -1]
                else:
                    raise ValueError(f"Unsupported observation shape: {obs.shape}")

            # Ensure float32 for normalization logic
            img_float = img_array.astype(np.float32)

            # Detect normalization range and denormalize properly
            vmin, vmax = float(np.min(img_float)), float(np.max(img_float))
            # If values in [-1, 1] assume our normalized pipeline
            if vmin >= -1.01 and vmax <= 1.01:
                img_uint8 = ((img_float * 128.0) + 128.0).clip(0, 255).astype(np.uint8)
            # If values in [0, 255], it's already uint-like
            elif vmin >= 0 and vmax <= 255:
                img_uint8 = img_float.clip(0, 255).astype(np.uint8)
            else:
                # Fallback: scale to 0-255
                if (vmax - vmin) < 1e-6:
                    img_uint8 = np.zeros_like(img_float, dtype=np.uint8)
                else:
                    img_uint8 = ((img_float - vmin) * (255.0 / (vmax - vmin))).clip(0, 255).astype(np.uint8)

            # Create PIL image and scale up to 336x336
            img = Image.fromarray(img_uint8, mode='L')
            img = img.resize((336, 336), Image.NEAREST)
            
            # Convert to RGB for adding text
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Try to use a good font, fall back to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
                font_large = font
            
            # Add HUD information
            hud_y = 10
            color = (0, 255, 0) if not done else (255, 255, 0)
            
            # Title
            draw.text((10, hud_y), "CARLA DRL Training", fill=(255, 255, 255), font=font_large)
            hud_y += 25
            
            # Stats
            draw.text((10, hud_y), f"Step: {step}", fill=color, font=font)
            hud_y += 18
            draw.text((10, hud_y), f"Reward: {reward:.4f}", fill=color, font=font)
            hud_y += 18
            draw.text((10, hud_y), f"Total: {total_reward:.2f}", fill=color, font=font)
            hud_y += 18
            if phi_t is not None and d_t is not None:
                # Show driving features if available
                draw.text((10, hud_y), f"phi: {phi_t:.3f} rad ({np.degrees(phi_t):.1f}°)", fill=(173, 216, 230), font=font)
                hud_y += 18
                draw.text((10, hud_y), f"d: {d_t:.3f} m", fill=(173, 216, 230), font=font)
                hud_y += 18
            
            status_color = (255, 0, 0) if done else (0, 255, 0)
            draw.text((10, hud_y), f"Status: {'DONE' if done else 'Active'}", fill=status_color, font=font)
            
            # Save frame
            filename = os.path.join(self.output_dir, f'frame_{self.frame_counter:04d}.png')
            img.save(filename)
            self.frame_counter += 1
            
            if self.frame_counter % 10 == 0:
                print(f"💾 Recorded {self.frame_counter} frames")
                
        except Exception as e:
            print(f"Error adding frame: {e}")
    
    def create_video(self, output_filename='training_video.mp4'):
        """Create MP4 video from saved frames using ffmpeg"""
        output_path = os.path.join(os.path.dirname(self.output_dir), output_filename)
        
        if self.frame_counter == 0:
            print("⚠️  No frames to create video")
            return None
        
        try:
            import subprocess
            
            # Check if ffmpeg is available
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True)
            if result.returncode != 0:
                print("⚠️  ffmpeg not installed. Video creation skipped.")
                print("   Install with: sudo apt-get install ffmpeg")
                return None
            
            # Create video with ffmpeg
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-framerate', str(self.fps),
                '-i', os.path.join(self.output_dir, 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',  # Quality (lower = better, 23 is good)
                output_path
            ]
            
            print(f"\n🎬 Creating video...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Video created: {output_path}")
                return output_path
            else:
                print(f"❌ Error creating video: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error creating video: {e}")
            return None
    
    def close(self):
        """Finalize recording"""
        print(f"\n✅ Total frames recorded: {self.frame_counter}")
        print(f"📁 Frames location: {self.output_dir}")
        
        # Optionally create video
        video_path = self.create_video()
        if video_path:
            print(f"🎥 You can watch the video with: vlc {video_path}")
