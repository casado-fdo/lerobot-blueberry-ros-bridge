
import base64
import io
import os
import time

import cv2
from ollama import Client
from pupil_labs.realtime_api.simple import discover_one_device
from utils import log_say
import sys

import logging
from pynput import keyboard


def init_keyboard_listener():
    # Dictionary to store the state of our keys
    events = {
        "enter": False,
        "esc": False,
        "last_number": None
    }

    def on_press(key):
        # Clear the line immediately to prevent echoed characters from showing
        print("\r\033[K", end='', flush=True) 
        try:
            # 1. Detect Numbers 1-4
            # pynput handles character keys through the .char attribute
            if hasattr(key, 'char') and key.char in ['1', '2', '3', '4']:
                events["last_number"] = int(key.char)

            # 2. Detect Arrows/ENTER/ESC
            elif key == keyboard.Key.right or key == keyboard.Key.enter:
                events["enter"] = True
            #elif key == keyboard.Key.left:
            #    print("Left arrow pressed.")
            elif key == keyboard.Key.esc:
                events["esc"] = True
                return False  # Returning False stops the listener thread

        except Exception as e:
            print(f"Error: {e}")

    # Start the listener in a non-blocking way
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events

class OllamaAssistant:
    """
    Eye-tracking visual assistant using Pupil Lab's Neon glasses together with Ollama for local vision models.
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

        self.key_listener, self.key_events = init_keyboard_listener()

        self.client = Client(host=ollama_host)
        self.ollama_host = ollama_host
        self.model = "llama3.2-vision"  # options: llava, qwen2.5vl:3b, qwen3-vl:4b, llama3.2-vision

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
            "in detail": (
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

    def assist(self):
        """
        Analyze the gaze point using the vision model and provide audio feedback.
        """
        try:
            log_say(f"Analyzing with mode: {self.mode}...")

            # Open a winddow displaying the frame
            cv2.imshow("Frame", self.matched)
            cv2.waitKey(1)

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
                think=False,
                keep_alive="0",
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
                    log_say(response_text)

            # Log the interaction
            self.session_count += 1
            print(
                f"✓ Response: {response_text}"
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
        print("  1        - Describe mode (8 words)")
        print("  2        - Dangers mode (identify risks)")
        print("  3        - Intention mode (infer user intent)")
        print("  4        - In Detail mode (full description)")
        print("  ->/ENTER - Analyze current frame") # right arrow or enter
        print("  ESC/q    - Quit")
        print("=" * 60)


    def run(self):
        """Main event loop - terminal based."""
        # Get the first frame from the camera and do nothing with it
        self.process_frame()
        self.print_menu()
        
        print("\nStarting eye-tracking loop...")
        print("(Frames are being captured but not analyzed until you press -> or ENTER)\n")

        try:
            while self.running:
                # Capture frame (but don't analyze it yet)
                if not self.process_frame():
                    continue
                # Check for user input
                if self.key_events.get("enter", False):
                    self.encode_image()
                    self.assist()
                    self.key_events["enter"] = False
                elif self.key_events.get("esc", False):
                    print("Quitting...")
                    self.running = False
                    self.key_events["esc"] = False
                elif self.key_events.get("last_number", None) is not None:
                    key = str(self.key_events["last_number"])

                    if key == "1":
                        self.mode = "describe"
                        print(f"Mode set to: Describe")
                    elif key == "2":
                        self.mode = "dangers"
                        print(f"Mode set to: Dangers")
                    elif key == "3":
                        self.mode = "intention"
                        print(f"Mode set to: Intention")    
                    elif key == "4":
                        self.mode = "in detail"
                        print(f"Mode set to: In Detail")
                    self.key_events["last_number"] = None
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