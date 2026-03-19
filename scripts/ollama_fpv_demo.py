
import base64
import io
import os
import time
import subprocess
from typing import Literal

import cv2
from dotenv import load_dotenv
from ollama import Client
from pupil_labs.realtime_api.simple import discover_one_device
from gtts import gTTS

load_dotenv()


class OllamaAssistant:
    """
    Eye-tracking visual assistant using Ollama with local vision models.
    Terminal-only interface (no GUI window).
    Free alternative to OpenAI's API - all processing happens locally.
    """

    def __init__(self, ollama_host: str = None):
        """
        Initialize the eye-tracking assistant.

        Args:
            ollama_host: URL where Ollama is running. 
                        If None, uses OLLAMA_HOST environment variable or defaults to http://localhost:11434
        """
        # Get Ollama host from environment variable or use provided value or default
        if ollama_host is None:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        self.device = None
        self.frame_target_width = 640
        self.frame_target_height = 480
        self.raw_width, self.raw_height = 1600, 1200  # native Neon resolution

        self.client = Client(host=ollama_host)
        self.ollama_host = ollama_host
        self.model = "qwen2.5vl:3b"  # options: moondream, llava, qwen2.5vl:3b, llama3.2-vision

        self.setup_prompts()
        self.mode = "describe"
        self.running = True

        self.session_count = 0
        self.initialise_device()


    def initialise_device(self):
        """Connect to Pupil Labs eye-tracking glasses."""
        print("Looking for the next best device...")
        self.device = discover_one_device(max_search_duration_seconds=10)
        if self.device is None:
            print("No device found.")
            raise SystemExit(-1)

        print(f"Connecting to {self.device}...")

    def setup_prompts(self):
        """Define analysis prompts for different modes."""
        self.prompts = {
            "base": (
                "You are a visual and communication aid for individuals with visual impairment "
                "(low vision) or communication difficulties. They are wearing eye-tracking glasses. "
                "An image is being sent with a red circle indicating where the wearer is looking. "
                "Do not describe the whole image unless explicitly asked. Be succinct and concise. "
                "Reply in English only."
            ),
            "describe": "In a couple of words (max. 8 words), say what the person is looking at.",
            "dangers": (
                "Briefly indicate if there is any risk posing danger to the person in the scene. "
                "Be succinct (max 20 words)."
            ),
            "intention": (
                "Given that the wearer may have mobility and speaking difficulties, "
                "briefly try to infer the wearer's intention based on what they are looking at. "
                "Maximum of 20 words."
            ),
            "in_detail": (
                "Describe the scene in detail, as if you were reading it aloud. "
                "Less than one minute of speaking (max 100 words)."
            ),
        }

    def process_frame(self):
        """Capture a frame from the eye-tracking device."""
        frame, gaze = self.device.receive_matched_scene_video_frame_and_gaze()
                
        # Draw gaze on the frame
        cv2.circle(
            frame.bgr_pixels,
            (int(gaze.x), int(gaze.y)),
            radius=45,
            color=(0, 0, 255),
            thickness=8,
        )

        # Crop to central region (remove 40% of margins)
        final_frame = frame.bgr_pixels[int(self.raw_height * 0.4):int(self.raw_height * 0.6), int(self.raw_width * 0.4):int(self.raw_width * 0.6)]
                
        # Resize frame to target resolution
        self.matched = cv2.resize(final_frame, (self.frame_target_width, self.frame_target_height), interpolation=cv2.INTER_LINEAR)

        return True

    def encode_image(self):
        """Encode the current frame as base64 for Ollama."""
        _, buffer = cv2.imencode(".jpg", self.matched)
        self.base64_frame = base64.b64encode(buffer).decode("utf-8")

    def say(self, text: str):
        """
        Text-to-speech using gTTS (Google Text-to-Speech) and mpg123.
        Works reliably without extra dependencies.
        
        Args:
            text: The text to speak
        """
        temp_file = "/tmp/speech_output.mp3"
        
        try:
            # Create the audio file using gTTS
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file)

            # Play the audio file using mpg123
            cmd = ["mpg123", temp_file]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        except FileNotFoundError:
            print("⚠️  mpg123 not found. Install with: apt-get install mpg123")
        except Exception as e:
            print(f"⚠️  Failed to play audio: {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def assist(self):
        """
        Analyze the gaze point using the vision model and provide audio feedback.
        """
        try:
            print(f"\n🔍 Analyzing with mode: {self.mode}...")

            # Prepare the full prompt
            full_prompt = (
                self.prompts["base"]
                + "\n\n"
                + self.prompts[self.mode]
            )

            # Call Ollama with vision model
            start_time = time.time()
            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                images=[self.base64_frame],
                stream=False,
            )
            inference_time = time.time() - start_time

            response_text = response["response"].strip()

            # Check if response is gibberish (many repeated characters)
            if len(response_text) > 0:
                # Count character repetitions
                unique_chars = len(set(response_text))
                if unique_chars < 3 and len(response_text) > 10:
                    print(f"⚠️  Gibberish detected in response, skipping TTS")
                    print(f"   Raw response: {response_text[:50]}...")
                else:
                    # Valid response - say it
                    self.say(response_text)

            # Log the interaction
            self.session_count += 1
            print(
                f"✓ Response: {response_text[:100]}"
            )
            print(
                f"  Inference time: {inference_time:.2f}s | Session interactions: {self.session_count}"
            )

        except Exception as e:
            print(f"✗ Error during analysis: {type(e).__name__}: {e}")

    def print_menu(self):
        """Print the terminal menu."""
        print("\n" + "=" * 60)
        print("Eye-Tracking Assistant (Ollama - Terminal Mode)")
        print("=" * 60)
        print("Commands:")
        print("  1 (or d) - Describe mode (8 words)")
        print("  2 (or s) - Dangers mode (identify risks)")
        print("  3 (or i) - Intention mode (infer user intent)")
        print("  4 (or f) - In Detail mode (full description)")
        print("  a        - Analyze current frame")
        print("  q        - Quit")
        print("=" * 60)

    def get_user_input(self):
        """Get user input in a non-blocking way."""
        import select
        import sys
        
        # Check if there's input available (Unix/Linux only)
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                return sys.stdin.read(1).lower()
            except:
                return None
        return None

    def run(self):
        """Main event loop - terminal based."""
        self.print_menu()
        
        print("\nStarting eye-tracking loop...")
        print("(Frames are being captured but not analyzed until you press 'a')\n")

        try:
            while self.running:
                # Capture frame (but don't analyze it yet)
                if not self.process_frame():
                    continue

                # Check for user input (non-blocking)
                user_input = self.get_user_input()
                
                if user_input:
                    print()  # New line after input
                    
                    if user_input == "q":
                        print("Quitting...")
                        self.running = False
                    elif user_input in ["1", "d"]:
                        self.mode = "describe"
                        print(f"Mode set to: Describe")
                    elif user_input in ["2", "s"]:
                        self.mode = "dangers"
                        print(f"Mode set to: Dangers")
                    elif user_input in ["3", "i"]:
                        self.mode = "intention"
                        print(f"Mode set to: Intention")
                    elif user_input in ["4", "f"]:
                        self.mode = "in_detail"
                        print(f"Mode set to: In Detail")
                    elif user_input == "a":
                        self.encode_image()
                        self.assist()
                    else:
                        # Ignore unknown commands silently (spaces, etc)
                        pass

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            print("\n" + "=" * 60)
            print(f"Stopping... Total interactions: {self.session_count}")
            print("=" * 60)
            if self.device:
                self.device.close()


if __name__ == "__main__":
    import sys
    
    # Get Ollama host from command line argument or environment variable
    ollama_host = None
    
    if len(sys.argv) > 1:
        ollama_host = sys.argv[1]
    else:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    eyes = OllamaAssistant(ollama_host)
    eyes.run()