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
from pygame import mixer
import matplotlib.pyplot as plt

from lerobot_robot_ros import BlueberryROS, BlueberryROSConfig
from utils import log_say, init_keyboard_listener


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
        min_confidence: float = 0.75,
        no_tts: bool = False,
    ):

        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = Client(host=self.ollama_host)
        self.model = model

        self.actions = actions or DEFAULT_ACTIONS
        self.action_executor = self.make_action_executor_map(self.actions)

        self.min_confidence = self.clamp(min_confidence, 0.0, 1.0)

        self.no_tts = no_tts

        self.key_listener, self.key_events = init_keyboard_listener()

        self.robot = BlueberryROS(BlueberryROSConfig())
        self.robot_connected = False

        self.user_name = os.getenv("USER_NAME", "Unknown")

        mixer.init()
        self.waiting_music = None

        # Initialise matplotlib plot for displaying frames
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.image_display = None

        self.state = "idle"  # idle, proposed, waiting_confirmation
        self.pending_proposal = None
        self.session_count = 0

        self.base_prompt = self.load_prompt("base_behaviour")

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))

    @staticmethod
    def make_action_executor_map(actions):
        return {
            "pick_bread_to_plate": GazeActionAssistant.execute_pick_bread_to_plate,
            "pick_cube_to_container": GazeActionAssistant.execute_pick_cube_to_container,
        }

    @staticmethod
    def execute_pick_bread_to_plate(robot: BlueberryROS):
        # Placeholder: real joint action should come from Robot action dictionary.
        log_say("Executing action: pick bread to plate. (This is a placeholder implementation.)", play_engine="gtts")
    
    @staticmethod
    def execute_pick_cube_to_container(robot: BlueberryROS):
        # Placeholder: real joint action should come from Robot action dictionary.
        log_say("Executing action: pick cube to container. (This is a placeholder implementation.)", play_engine="gtts")

    def get_time_of_day(self):
        return datetime.now().strftime("%H:%M")

    def load_prompt(self, prompt_id):
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
        print(f"Loading prompt '{prompt_id}' with context: {context_vars}")

        # Read the prompt file
        with open(f"scripts/prompts/{prompt_id}.txt", "r", encoding="utf-8") as f:
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
            log_say("Robot connected. Ready for gaze-assisted actions.", play_engine="gtts")
        except Exception as e:
            self.robot_connected = False
            print(f"Robot connection warning: {e}")
            log_say("Could not connect to robot; action execution will be disabled.", play_engine="gtts")

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
                keep_alive=0,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        text = response.get("response", "").strip()
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

        if action_id is None:
            return False

        if confidence >= self.min_confidence:
            return True

        return False

    def execute_action(self, action_id):
        if not action_id:
            return False, "No action id to execute."

        if action_id not in self.actions:
            return False, "Action not recognized."

        executor = self.action_executor.get(action_id)
        if executor is None:
            return False, "Action has no executor mapped."

        if not self.robot_connected:
            return False, "Robot is not connected."

        try:
            executor(self.robot)
            return True, f"Executed {action_id} successfully."
        except Exception as e:
            return False, f"Failed to execute {action_id}: {e}"

    def run(self):
        self.connect_robot()

        try:
            log_say("Gaze assistant ready. Look at an object and press Enter to analyze.", play_engine="gtts")
        except Exception:
            # TTS may fail in headless environment
            print("Speak: gaze assistant ready")

        print("Press Enter to suggest an action, ESC to quit, 1/2/3 to cycle modes (currently fixed)")

        try:
            while True:
                # No blocking UI, just event-driven keyboard state.
                if self.key_events.get("enter"):
                    self.key_events["enter"] = False
                    
                    user_view = self.get_robot_frame()

                    if self.state == "idle":
                        encoded_user_view = self.encode_image(user_view)
                        vlm_output = self.query_vlm(self.base_prompt, user_view, encoded_user_view)

                        # Speak the message no matter what
                        if not self.no_tts:
                            log_say(vlm_output["message"], play_engine="gtts")
                        else:
                            print(vlm_output["message"])

                        self.session_count += 1

                        if self.apply_execution_policy(vlm_output):
                            self.pending_proposal = vlm_output
                            self.state = "waiting_confirmation"
                            msg = (
                                f"I propose action {vlm_output['action_id']} "
                                f"(confidence={vlm_output['confidence']:.2f}). Press Enter to confirm or ESC to cancel."
                            )
                            if not self.no_tts:
                                log_say(msg, play_engine="gtts")
                            else:
                                print(msg)
                        else:
                            self.pending_proposal = None
                            self.state = "idle"
                            print("No eligible action proposal. Returning to idle.")

                    elif self.state == "waiting_confirmation":
                        assert self.pending_proposal
                        action_id = self.pending_proposal.get("action_id")
                        success, details = self.execute_action(action_id)
                        if not self.no_tts:
                            log_say(details, play_engine="gtts")
                        else:
                            print(details)
                        self.pending_proposal = None
                        self.state = "idle"

                if self.key_events.get("esc"):
                    self.key_events["esc"] = False
                    if self.state == "waiting_confirmation":
                        self.pending_proposal = None
                        self.state = "idle"
                        log_say("Action cancelled. Back to idle.", play_engine="gtts")
                        continue

                    log_say("Quitting gaze assistant.", play_engine="gtts")
                    break

                # Optional: mode selection by number keys
                if self.key_events.get("last_number") is not None:
                    n = self.key_events["last_number"]
                    self.key_events["last_number"] = None
                    print(f"Action selection number pressed (not used): {n}")

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: exiting")

        finally:
            if self.robot_connected:
                try:
                    self.robot.disconnect()
                except Exception:
                    pass

            print(f"Session interactions: {self.session_count}")


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
    parser.add_argument("--model", type=str, default="qwen3.5:9b")
    parser.add_argument("--min_confidence", type=float, default=0.75)
    parser.add_argument("--no_tts", action="store_true")
    args = parser.parse_args()

    actions = DEFAULT_ACTIONS
    if args.actions_json:
        actions = load_actions_from_file(args.actions_json)

    assistant = GazeActionAssistant(
        ollama_host=args.ollama_host,
        actions=actions,
        model=args.model,
        min_confidence=args.min_confidence,
        no_tts=args.no_tts,
    )

    assistant.run()


if __name__ == "__main__":
    main()
