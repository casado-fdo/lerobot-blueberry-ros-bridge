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
import matplotlib.pyplot as plt
from pygame import mixer

USER_NAME = "Fernando"
USER_GENDER = "male"
DAY_OF_WEEK = "Monday"
CURRENT_TIME = time.strftime("%H:%M")

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
        self.model = "llama3.2-vision"  # options: llava:7b-v1.6-mistral-q2_K, qwen2.5vl:3b, qwen3-vl:2b, qwen3-vl:4b, llama3.2-vision, moondream

        self.setup_prompts()
        self.mode = "describe"
        self.running = True

        self.session_count = 0
        self.initialise_device()

        # Initialise pygame mixer for audio
        mixer.init()
        self.waiting_music = mixer.Sound("media/waiting_music.mp3")

        # Initialise matplotlib plot for displaying frames
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.image_display = None


    def initialise_device(self):
        """Connect to Pupil Labs eye-tracking glasses."""
        print("Looking for the next best device...")
        self.device = discover_one_device(max_search_duration_seconds=10)
        if self.device is None:
            print("No device found.")
            raise SystemExit(-1)

        print(f"Connecting to {self.device}...")

    def update_plot(self, img):
        
        # 1. Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. If it's the first time, create the display object
        if self.image_display is None:
            self.image_display = self.ax.imshow(img_rgb)
            self.ax.axis('off')
        else:
            # 3. Just update the data (much faster, prevents grey screen)
            self.image_display.set_data(img_rgb)
        
        # 4. Force a draw and a tiny pause to let Ubuntu paint the window
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def read_prompt(self, prompt_name):
        """Read a prompt from a file."""
        with open(f"scripts/prompts/{prompt_name}.txt", "r") as f:
            return f.read()

    def setup_prompts(self):
        """Define analysis prompts for different modes."""
        self.prompts = {
            "base": (
                self.read_prompt("base")
            ),
            "describe": (
                self.read_prompt("describe")
            ),
            "intention": (
                self.read_prompt("intention")
            ),
            "in detail": (
                self.read_prompt("in_detail")
            ),
            "greetings": (
                self.read_prompt("greetings")
            ),
            "additional_context": (
                f"Today is {DAY_OF_WEEK} and the current time is {CURRENT_TIME}. "
                f"My name is Blueberry, and I am your bimanual robot wheelchair. "
                f"Your name is {USER_NAME}, and you identify as a {USER_GENDER}."
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

        # Crop to central region (remove 25% of margins)
        final_frame = frame.bgr_pixels[int(self.raw_height * 0.25):int(self.raw_height * 0.75), int(self.raw_width * 0.25):int(self.raw_width * 0.75)]
                
        # Resize frame to target resolution
        self.matched = cv2.resize(final_frame, (self.frame_target_width, self.frame_target_height), interpolation=cv2.INTER_LINEAR)

        return True

    def encode_image(self):
        """Encode the current frame as base64 for Ollama."""
        _, buffer = cv2.imencode(".jpg", self.matched)
        self.base64_frame = base64.b64encode(buffer).decode("utf-8")

    def call_ollama(self, prompt, image=None):
        """Call Ollama with the given prompt."""
        return self.client.generate(
            model=self.model,
            prompt=prompt,
            images=[image] if image else None,
            stream=False,
            keep_alive="0",
        )

    def assist(self):
        try:
            #log_say(f"Analyzing scene...", play_engine="kokoro")
            self.update_plot(self.matched)

            # Start the music on a loop (-1 means infinite loop)
            music_channel = self.waiting_music.play(loops=-1)
            music_channel.set_volume(0.7)

            # Prepare the prompt
            full_prompt = self.prompts["base"] + "\n\n" + self.prompts["additional_context"] + "\n\n" + self.prompts[self.mode]

            # Call Ollama (The music plays while the main thread is busy here)
            start_time = time.time()
            response = self.call_ollama(full_prompt, self.base64_frame)
            inference_time = time.time() - start_time

            response_text = response["response"].strip()

            # Stop the music the moment the model returns a result
            music_channel.fadeout(1000) # Smooth 1s fade out

            # Check if response is gibberish (many repeated characters)
            if len(response_text) > 0:
                # Count character repetitions
                unique_chars = len(set(response_text))
                if unique_chars < 3 and len(response_text) > 10:
                    print(f"⚠️  Gibberish detected in response, skipping TTS")
                    print(f"   Raw response: {response_text[:50]}...")
                else:
                    # Valid response - say it
                    log_say(response_text, play_engine="kokoro")

            # Log the interaction
            self.session_count += 1
            print(
                f"  Inference time: {inference_time:.2f}s | Session interactions: {self.session_count}"
            )

        except Exception as e:
            mixer.stop() # Safety stop if the code crashes
            print(f"✗ Error during analysis: {e}")

    def print_menu(self):
        """Print the terminal menu."""
        print("\n" + "=" * 60)
        print("Eye-Tracking Assistant (Ollama - Terminal Mode)")
        print("=" * 60)
        print("Commands:")
        print("  1        - Describe mode (8 words)")
        print("  2        - Intention mode (infer user intent)")
        print("  3        - In Detail mode (full description)")
        print("  ->/ENTER - Analyze current frame") # right arrow or enter
        print("  ESC      - Quit")
        print("=" * 60)


    def run(self):
        """Main event loop - terminal based."""
        
        init_prompt = self.prompts["base"] + "\n" + self.prompts["additional_context"] + "\n" + self.prompts["greetings"]
        greetings = self.call_ollama(init_prompt)["response"].strip()
        log_say(greetings, play_engine="kokoro")

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
                        self.mode = "intention"
                        print(f"Mode set to: Intention")
                    elif key == "3":
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