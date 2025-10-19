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
        """Add a frame to the video"""
        try:
            # Take the last frame if it's a stack
            if len(observation.shape) == 3:
                frame = observation[:, :, -1]
            else:
                frame = observation
            
            # Convert to uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Scale up to 336x336 for better visibility
            img = Image.fromarray(frame, mode='L')
            img = img.resize((336, 336), Image.NEAREST)
            
            # Convert to RGB for adding text
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Try to use a good font, fall back to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
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
