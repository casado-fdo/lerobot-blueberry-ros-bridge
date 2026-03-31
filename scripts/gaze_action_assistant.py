import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from ollama import Client
import matplotlib.pyplot as plt

from lerobot_robot_ros import BlueberryROS, BlueberryROSConfig
from io_manager import IOManager


# Default action set (action_id -> description, constraints optional)
DEFAULT_ACTIONS = {
    "pick_bread_to_plate": {
        "action_id": "pick_bread_to_plate",
        "description": "Pick the sandwich bread and place it on the plate.",
        "constraints": "Only use when bread is in reachable area and plate is present."
    },
    "pick_cube_to_container": {
        "action_id": "pick_cube_to_container",
        "description": "Pick the cube and place it into the container.",
        "constraints": "Only use when cube and container are visible and unobstructed."
    },
}


class GazeActionAssistant:

    def __init__(
        self,
        ollama_host: str = None,
        actions: dict = None,
        model: str = "qwen3.5:9b",
        min_confidence: float = 0.5,
        use_tts: bool = True,
        use_audio_notifications: bool = True,
    ):

        # I/O setup
        self.use_tts = use_tts
        self.use_audio_notifications = use_audio_notifications
        self.tts_engine = "kokoro" if self.use_tts else "none"
        self.io = IOManager(audio_enabled=self.use_audio_notifications, tts_engine=self.tts_engine)
        self.io.play_booting_music()


        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = Client(host=self.ollama_host)
        self.model = model

        self.actions = actions or DEFAULT_ACTIONS

        self.min_confidence = self.clamp(min_confidence, 0.0, 1.0)

        self.robot = BlueberryROS(BlueberryROSConfig())
        self.robot_connected = False

        self.user_name = os.getenv("USER_NAME", "Unknown")


        # Initialise matplotlib plot for displaying frames
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.image_display = None

        self.state = "idle"  # idle, waiting_confirmation
        self.pending_proposal = None
        self.session_count = 0
        self.esc_press_count = 0
        self.last_esc_time = 0

        self.base_prompt = self.load_base_prompt()
        self.greetings_prompt = self.load_greetings_prompt()

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))

    def get_time_of_day(self):
        return datetime.now().strftime("%H:%M")

    def load_greetings_prompt(self):
        # Define the variables to inject
        context_vars = {
            "time_of_day": self.get_time_of_day(),
            "user_name": self.user_name or 'Unknown',
        }
        # Debug context
        print(f"Loading prompt 'greetings' with context: {context_vars}")

        # Read the prompt file
        with open(f"scripts/prompts/greetings.txt", "r", encoding="utf-8") as f:
            template = f.read()

        # Perform substitution (matches {variable_name}) for context variables
        formatted_prompt = template.format(**context_vars)
        return formatted_prompt

    def load_base_prompt(self):
        # Define the variables to inject
        action_set =  "\n".join(
            [f"- {a['action_id']}: {a['description']}" for a in self.actions.values()]
        )
        action_ids = [a['action_id'] for a in self.actions.values()]
        context_vars = {
            "time_of_day": self.get_time_of_day(),
            "user_name": self.user_name or 'Unknown',
            "action_set": action_set,
            "action_ids": action_ids,
        }
        # Debug context
        print(f"Loading prompt 'base_behaviour' with context: {context_vars}")

        # Read the prompt file
        with open(f"scripts/prompts/base_behaviour.txt", "r", encoding="utf-8") as f:
            template = f.read()

        # Perform substitution (matches {variable_name}) for context variables
        formatted_prompt = template.format(**context_vars)
        return formatted_prompt

    def get_robot_frame(self):
        if not self.robot_connected:
            raise RuntimeError("Robot is not connected; cannot fetch camera frame.")

        obs = self.robot.get_observation()
        frame = obs.get("user") if isinstance(obs, dict) else None
        if frame is None:
            raise RuntimeError("No 'user' camera frame available from robot observation.")
        if not isinstance(frame, np.ndarray):
            raise RuntimeError("Robot 'user' camera frame is not a numpy array.")
        # Frame is rgb but we want bgr for opencv display and encoding
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def connect_robot(self):
        try:
            self.robot.connect()
            self.robot_connected = True
            self.io.log("Robot connected. Ready for gaze-assisted actions.", speak=False)
        except Exception as e:
            self.robot_connected = False
            self.io.log(f"Failed to connect to robot: {e}", speak=False)

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
                    model=self.model,
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
                model=self.model,
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
            return self.default_vlm_output("Empty model response", "none", 0.0)

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
                return self.default_vlm_output(msg, "none", 0.0)

        # Validate required fields
        obj = payload.get("object")
        intent = payload.get("intent")
        action_id = payload.get("action_id")
        confidence = payload.get("confidence")
        message = payload.get("message")
        reasoning = payload.get("reasoning", None)

        if action_id is None or action_id not in self.actions:
            action_id = "none_detected"

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        confidence = self.clamp(confidence, 0.0, 1.0)

        if message is None or not isinstance(message, str):
            message = "I can’t generate a safe natural language message, so no action will run."

        result = {
            "object": obj,
            "intent": intent,
            "action_id": action_id,
            "confidence": confidence,
            "message": message,
            "reasoning": reasoning,
        }

        return result

    def default_vlm_output(self, message, confidence):
        return {
            "intent": None,
            "object": None,
            "action_id": None,
            "confidence": confidence,
            "message": message,
            "reasoning": None,
        }

    def apply_execution_policy(self, vlm_output):
        action_id = vlm_output.get("action_id")
        confidence = vlm_output.get("confidence", 0.0)

        if action_id is None or action_id == 'none_detected':
            return False

        if confidence >= self.min_confidence:
            return True

        return False

    def execute_action(self, action_id):
        if not action_id:
            return False, "No action id to execute."

        if action_id not in self.actions:
            return False, "Action not recognized."

        task_description = self.actions[action_id].get("description", None)
        if task_description is None:
            return False, "Action has no description available."

        if not self.robot_connected:
            return False, "Robot is not connected."

        try:
            # Placeholder for now
            print("PLACEHOLDER: ACTION EXECUTED.")
            return True, f"Executed {action_id} successfully."
        except Exception as e:
            return False, f"Failed to execute {action_id}: {e}"

    def run(self):
        self.connect_robot()

        if self.robot_connected:
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

                if keyboard_events.get("enter") and not keyboard_events.get("esc"):                    
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
                                f"I propose action {vlm_output['action_id']} "
                                f"(confidence={vlm_output['confidence']:.2f})."
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
                        success, details = self.execute_action(action_id)
                        self.io.log(details, speak=False)
                        self.pending_proposal = None
                        self.state = "idle"

                if keyboard_events.get("esc") and not keyboard_events.get("enter"):
                    current_time = time.time()
                    if current_time - self.last_esc_time > 2.0:
                        self.esc_press_count = 0
                    
                    self.esc_press_count += 1
                    self.last_esc_time = current_time
                    
                    if self.state == "waiting_confirmation":
                        self.pending_proposal = None
                        self.state = "idle"
                        self.io.notify(self.io.UPDATE, "preset:action_cancelled", speak=self.use_tts)
                        self.io.notify(self.io.IDLE)
                        self.esc_press_count = 0  # Reset ESC count after cancelling
                        continue

                    # Only exit after double ESC press
                    if self.esc_press_count >= 2:
                        self.io.notify(self.io.UPDATE, "preset:goodbye", speak=self.use_tts)
                        self.io.play_logout_music()
                        time.sleep(2.0) # Placeholder for now
                        break
                    
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: exiting")

        finally:
            self.io.stop_logout_music()
            time.sleep(2.0) # Placeholder for now
            if self.robot_connected:
                try:
                    self.robot.disconnect()
                except Exception:
                    pass

            


def load_actions_from_file(file_path):
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


def main():
    parser = argparse.ArgumentParser(description="Gaze-based assistive action assistant.")
    parser.add_argument("--ollama_host", type=str, default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--actions_json", type=str, default=None, help="Optional JSON file defining actions.")
    parser.add_argument("--model", type=str, default="qwen3.5:9b") #"llama3.2-vision:11b") # "qwen3.5:4b")
    parser.add_argument("--min_confidence", type=float, default=0.5)
    parser.add_argument("--use_tts", type=bool, default=True)
    parser.add_argument("--use_audio_notifications", type=bool, default=True)
    args = parser.parse_args()

    actions = DEFAULT_ACTIONS
    if args.actions_json:
        actions = load_actions_from_file(args.actions_json)

    assistant = GazeActionAssistant(
        ollama_host=args.ollama_host,
        actions=actions,
        model=args.model,
        min_confidence=args.min_confidence,
        use_tts=args.use_tts,
    )

    assistant.run()

if __name__ == "__main__":
    main()
