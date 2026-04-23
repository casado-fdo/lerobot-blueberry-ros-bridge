import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from ollama import Client
import matplotlib.pyplot as plt
from lerobot_robot_ros import BlueberryInference
from io_manager import IOManager


class GazeActionAssistant:

    def __init__(
        self,
        hf_username: str,
        hf_policy_id: str,
        hf_dataset_id: str,
        fps: int,
        episode_time_sec: int,
        reset_time_sec: int,
        ollama_host: str,
        actions_json: str,
        ollama_model: str,
        use_tts: bool = True,
        use_audio_notifications: bool = True,
        user_name: str = "Unknown",
    ):

        # I/O setup
        self.use_tts = use_tts
        self.use_audio_notifications = use_audio_notifications
        self.tts_engine = "kokoro" if self.use_tts else "none"
        self.io = IOManager(audio_enabled=self.use_audio_notifications, tts_engine=self.tts_engine)
        self.io.play_booting_music()

        # Ollama setup
        self.ollama_host = ollama_host
        self.client = Client(host=self.ollama_host)
        self.ollama_model = ollama_model
        self.user_name = user_name
        self.actions = self.load_actions_from_file(actions_json)
        
        # Robot interface setup
        self.blueberry_infer = BlueberryInference(hf_username, hf_policy_id, hf_dataset_id, fps)
        self.episode_time_sec = episode_time_sec
        self.reset_time_sec = reset_time_sec

        # Initialise matplotlib plot for displaying frames (debug)
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.image_display = None

        self.state = "idle"  # idle, waiting_confirmation
        self.pending_proposal = None
        self.session_count = 0
        self.esc_press_count = 0
        self.last_cancel_time = 0

        self.base_prompt = self.load_fpv_assistance_prompt()
        self.greetings_prompt = self.load_greetings_prompt()

    def get_time_of_day(self):
        return datetime.now(ZoneInfo("Europe/London")).strftime("%H:%M")

    def load_actions_from_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Action definition file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        structured = {}
        for k, v in data.items():
            if isinstance(v, str):
                structured[k] = {"action_id": k, "description": v}
            elif isinstance(v, dict):
                structured[k] = {"action_id": k, "description": v.get("description", ""), "constraints": v.get("constraints")}
            else:
                raise ValueError(f"Unsupported action format for {k}: {v}")

        return structured

    def load_greetings_prompt(self):
        # Define the variables to inject
        context_vars = {
            "time_of_day": self.get_time_of_day(),
            "user_name": self.user_name or 'Unknown',
        }
        # Debug context
        print(f"Loading prompt 'greetings' with context: {context_vars}")

        # Read the prompt file
        with open(f"prompts/greetings.txt", "r", encoding="utf-8") as f:
            template = f.read()

        # Perform substitution (matches {variable_name}) for context variables
        formatted_prompt = template.format(**context_vars)
        return formatted_prompt

    def load_fpv_assistance_prompt(self):
        # Define the variables to inject
        action_set =  "\n".join(
            [f"- {a['action_id']}: {a['description']}" for a in self.actions.values()]
        )
        action_ids = [a['action_id'] for a in self.actions.values()]
        context_vars = {
            "user_name": self.user_name or 'Unknown',
            "action_set": action_set,
            "action_ids": action_ids,
        }
        # Debug context
        print(f"Loading prompt 'fpv_assistance' with context: {context_vars}")

        # Read the prompt file
        with open(f"prompts/fpv_assistance.txt", "r", encoding="utf-8") as f:
            template = f.read()

        # Perform substitution (matches {variable_name}) for context variables
        formatted_prompt = template.format(**context_vars)
        return formatted_prompt

    def get_robot_frame(self):
        try:
            frame = self.blueberry_infer.get_latest_fpv_frame(desired_height=240, desired_width=320, display_gaze=True)
        except Exception as e:
            raise RuntimeError(f"Failed to get robot frame: {e}")
        # Frame is rgb but we want bgr for opencv display and encoding
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

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

    def encode_image(self, frame):
        if frame is None:
            raise RuntimeError("No frame to encode yet")
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def generate_greetings(self):
        try:
            response = self.client.generate(
                    model=self.ollama_model,
                    prompt=self.greetings_prompt,
                    stream=False,
                    think=False,
                    #keep_alive=0,
                )
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")
        
        text = response.get("response", "").strip()
        return text

    def query_vlm(self, prompt, user_view=None, encoded_user_view=None):
        try:
            if user_view is not None:
                self.update_plot(user_view)
            response = self.client.generate(
                model=self.ollama_model,
                prompt=prompt,
                images=[encoded_user_view] if encoded_user_view else None,
                stream=False,
                think=False,
                format="json",
                #keep_alive=0,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        text = response.get("response", "").strip()
        print(f"Prompt tokens: {response.get('prompt_eval_count', 'N/A')}")
        print(f"Response tokens: {response.get('eval_count', 'N/A')}")
        total_duration = response.get('total_duration', 'N/A')
        if total_duration != 'N/A':
            print(f"Total duration: {total_duration/1e9:.3f} s")
        print(f"VLM response: \n {text}")
        return self.parse_vlm_response(text)

    def parse_vlm_response(self, text: str):
        if not text:
            return self.default_vlm_output("Empty model response")

        # Extract JSON from free text if there is surrounding text.
        try:
            json_start = text.index("{")
            json_end = text.rindex("}") + 1
            json_text = text[json_start:json_end]
            payload = json.loads(json_text)
        except Exception:
            # fallback: try whole text as JSON
            try:
                payload = json.loads(text)
            except Exception:
                msg = "I couldn't parse your response correctly, so I will not propose an action."
                return self.default_vlm_output(msg)

        # Validate required fields
        obj = payload.get("object")
        intent = payload.get("intent")
        action_id = payload.get("action_id")
        message = payload.get("message")
        reasoning = payload.get("reasoning", None)

        if action_id is None or action_id not in self.actions:
            action_id = "none"

        if message is None or not isinstance(message, str):
            message = "I can’t generate a safe natural language message, so no action will run."

        result = {
            "object": obj,
            "intent": intent,
            "action_id": action_id,
            "message": message,
            "reasoning": reasoning,
        }

        return result

    def default_vlm_output(self, message):
        return {
            "intent": None,
            "object": None,
            "action_id": None,
            "message": message,
            "reasoning": None,
        }

    def apply_execution_policy(self, vlm_output):
        action_id = vlm_output.get("action_id")

        if action_id is None or action_id == 'none':
            return False

        return True

    def execute_action(self, keyboard_events, action_id):
        if not action_id:
            return False, "No action id to execute."

        if action_id not in self.actions:
            return False, "Action not recognized."

        task_description = self.actions[action_id].get("description", None)
        if task_description is None:
            return False, "Action has no description available."

        if not self.blueberry_infer.is_connected():
            return False, "Robot is not connected."

        try:
            self.io.log(f"Executing action {action_id} with task description: {task_description}", speak=False)
            self.io.reset_keyboard_events()
            self.blueberry_infer.run_inference_loop(self.io.get_keyboard_events(), self.episode_time_sec, task_description)
            self.io.reset_keyboard_events()
            self.io.log("Resetting environment...", speak=False)
            time.sleep(self.reset_time_sec)
            return True, f"Executed {action_id} successfully."
        except Exception as e:
            return False, f"Failed to execute {action_id}: {e}"

    def run(self):
        if self.blueberry_infer.is_connected():
            summary_msg = "Robot connected and ready for assistance:\n"
            summary_msg += self.blueberry_infer.get_summary()
            self.io.log(summary_msg, speak=False)
            greetings = self.generate_greetings()
            self.io.stop_booting_music()
            time.sleep(2.0)
            self.io.notify(self.io.UPDATE, greetings, speak=self.use_tts)
            self.io.notify(self.io.IDLE)
        else:
            self.io.notify(self.io.FAIL, "preset:error_booting", speak=self.use_tts)
            return

        self.io.log("Press the Assistance button to suggest an action, or the Exit button to quit", speak=False)

        try:
            while True:
                keyboard_events = self.io.get_keyboard_events()

                if keyboard_events.get("assist") and not keyboard_events.get("cancel"): 
                    self.io.reset_keyboard_events()                   
                    user_view = self.get_robot_frame()

                    if self.state == "idle":
                        self.io.play_processing_music()
                        encoded_user_view = self.encode_image(user_view)
                        vlm_output = self.query_vlm(self.base_prompt, user_view, encoded_user_view)

                        self.session_count += 1

                        if self.apply_execution_policy(vlm_output):
                            self.io.notify(self.io.SUCCESS, vlm_output["message"], speak=self.use_tts)
                            self.pending_proposal = vlm_output
                            self.state = "waiting_confirmation"
                            log_msg = (
                                f"I propose action {vlm_output['action_id']}."
                            )
                            self.io.log(log_msg, speak=False)
                            self.io.log("Press the Assistance button to confirm or the Exit button to cancel.", speak=False)
                            self.io.notify(self.io.IDLE)
                        else:
                            self.io.notify(self.io.FAIL, vlm_output["message"], speak=self.use_tts)
                            self.io.notify(self.io.IDLE)
                            self.pending_proposal = None
                            self.state = "idle"

                    elif self.state == "waiting_confirmation":
                        assert self.pending_proposal
                        action_id = self.pending_proposal.get("action_id")
                        self.io.notify(self.io.UPDATE)
                        success, details = self.execute_action(keyboard_events, action_id)
                        self.io.log(details, speak=False)
                        self.pending_proposal = None
                        self.state = "idle"

                elif keyboard_events.get("cancel") and not keyboard_events.get("assist"):
                    self.io.reset_keyboard_events()
                    current_time = time.time()
                    if current_time - self.last_cancel_time > 2.0:
                        self.cancel_press_count = 0
                    
                    self.cancel_press_count += 1
                    self.last_cancel_time = current_time
                    
                    if self.state == "waiting_confirmation":
                        self.pending_proposal = None
                        self.state = "idle"
                        self.io.notify(self.io.UPDATE, "preset:action_cancelled", speak=self.use_tts)
                        self.io.notify(self.io.IDLE)
                        self.cancel_press_count = 0  # Reset cancel count after cancelling
                        continue

                    # Only exit after double cancel press
                    if self.cancel_press_count >= 2:
                        self.io.notify(self.io.UPDATE, "preset:goodbye", speak=self.use_tts)
                        self.io.play_logout_music()
                        break
                    
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: exiting")

        finally:
            self.io.stop_logout_music()
            if self.blueberry_infer.is_connected():
                try:
                    self.blueberry_infer.disconnect()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Gaze-based assistive action assistant.")
    # OLLAMA
    parser.add_argument("--ollama_host", type=str, default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--actions_json", type=str, default="prompts/robot_actions.json")
    parser.add_argument("--ollama_model", type=str, default=os.getenv("OLLAMA_MODEL", "qwen3.5:9b")) #"llama3.2-vision:11b") # "qwen3.5:4b")
    # HRI
    parser.add_argument("--use_tts", type=bool, default=os.getenv("USE_TTS", "true").lower() == "true")
    parser.add_argument("--use_audio_notifications", type=bool, default=os.getenv("USE_AUDIO_NOTIFICATIONS", "true").lower() == "true")
    parser.add_argument("--user_name", type=str, default=os.getenv("USER_NAME", "Unknown"))
    # LeRobot
    parser.add_argument("--policy_id", type=str, default=os.getenv("HUGGINGFACE_MODEL_NAME", "your-policy-id-here"), help="HuggingFace policy name to evaluate")
    parser.add_argument("--dataset_id", type=str, default=None, help="HuggingFace dataset name to use for stats")
    parser.add_argument("--fps", type=int, default=int(os.getenv("RECORDING_FPS", "30")), help="Frames per second for evaluation (default: 30)")
    parser.add_argument("--episode_time_sec", type=int, default=int(os.getenv("RECORDING_EPISODE_TIME_SEC", "10")), help="Duration of each episode in seconds (default: 10)")
    parser.add_argument("--reset_time_sec", type=int, default=int(os.getenv("RECORDING_RESET_TIME_SEC", "5")), help="Time to reset between episodes in seconds (default: 5)")
    parser.add_argument("--hf_username", type=str, default=os.getenv("HUGGINGFACE_USERNAME", "your-username-here"), help="HuggingFace username (default: your-username-here)")
    args = parser.parse_args()

    assistant = GazeActionAssistant(
        hf_username=args.hf_username,
        hf_policy_id=args.policy_id,
        hf_dataset_id=args.dataset_id,
        fps=args.fps,
        episode_time_sec=args.episode_time_sec,
        reset_time_sec=args.reset_time_sec,
        ollama_host=args.ollama_host,
        actions_json=args.actions_json,
        ollama_model=args.ollama_model,
        use_tts=args.use_tts,
        use_audio_notifications=args.use_audio_notifications,
        user_name=args.user_name,
    )

    assistant.run()

if __name__ == "__main__":
    main()
