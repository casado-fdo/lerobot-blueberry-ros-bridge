import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
import cv2
from ollama import Client
from pupil_labs.realtime_api.simple import discover_one_device
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
    MATCH_TYPES = {"exact", "adapted", "none"}

    def __init__(
        self,
        ollama_host: str = None,
        actions: dict = None,
        model: str = "qwen3.5:9b",
        min_exact_confidence: float = 0.75,
        min_adapted_confidence: float = 0.5,
        no_tts: bool = False,
    ):

        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = Client(host=self.ollama_host)
        self.model = model

        self.actions = actions or DEFAULT_ACTIONS
        self.action_executor = self.make_action_executor_map(self.actions)

        self.min_exact_confidence = self.clamp(min_exact_confidence, 0.0, 1.0)
        self.min_adapted_confidence = self.clamp(min_adapted_confidence, 0.0, 1.0)

        self.no_tts = no_tts

        self.device = None
        self.frame_target_width = 640
        self.frame_target_height = 480
        self.raw_width, self.raw_height = 1600, 1200
        self.matched = None
        self.base64_frame = None

        self.key_listener, self.key_events = init_keyboard_listener()

        self.robot = BlueberryROS(BlueberryROSConfig())
        self.robot_connected = False

        self.user_name = os.getenv("USER_NAME", "Unknown")
        self.user_gender = os.getenv("USER_GENDER", "Unknown")

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
        context_vars = {
            "time_of_day": self.get_time_of_day(),
            "user_name": self.user_name or 'Unknown',
            "action_set": action_set,
        }
        # Debug context
        print(f"Loading prompt '{prompt_id}' with context: {context_vars}")

        # Read the prompt file
        with open(f"scripts/prompts/{prompt_id}.txt", "r", encoding="utf-8") as f:
            template = f.read()

        # Perform substitution (matches {variable_name}) for context variables
        formatted_prompt = template.format(**context_vars)
        print(f"Formatted prompt: {formatted_prompt}")
        return formatted_prompt

    def initialise_device(self):
        print("Looking for Pupil Labs Neon device...")
        self.device = discover_one_device(max_search_duration_seconds=10)
        if self.device is None:
            raise SystemExit("Could not find eye-tracking device.")
        print(f"Connected to device: {self.device}")

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

    def process_frame(self):
        frame, gaze = self.device.receive_matched_scene_video_frame_and_gaze()

        cv2.circle(
            frame.bgr_pixels,
            (int(gaze.x), int(gaze.y)),
            radius=45,
            color=(0, 0, 255),
            thickness=8,
        )

        h, w = self.raw_height, self.raw_width
        cy1, cy2 = int(h * 0.25), int(h * 0.75)
        cx1, cx2 = int(w * 0.25), int(w * 0.75)

        roi = frame.bgr_pixels[cy1:cy2, cx1:cx2]
        self.matched = cv2.resize(roi, (self.frame_target_width, self.frame_target_height), interpolation=cv2.INTER_LINEAR)

        return True

    def encode_image(self):
        if self.matched is None:
            raise RuntimeError("No frame to encode yet")
        _, buffer = cv2.imencode('.jpg', self.matched)
        self.base64_frame = base64.b64encode(buffer).decode('utf-8')

    def query_vlm(self):
        if self.base64_frame is None:
            raise RuntimeError("No encoded frame")
        try:
            self.update_plot(self.matched)
            response = self.client.generate(
                model=self.model,
                prompt=self.base_prompt,
                images=[self.base64_frame],
                stream=False,
                think=False,
                format="json",
                keep_alive=0,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        text = response.get("response", "").strip()
        print(f"RAW VLM response: {text}")
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
        intent = payload.get("intent")
        obj = payload.get("object")
        action_id = payload.get("action_id")
        match_type = payload.get("match_type")
        confidence = payload.get("confidence")
        message = payload.get("message")
        reasoning = payload.get("reasoning", None)

        if match_type not in self.MATCH_TYPES:
            message = "I can’t map the match type safely. No action will be performed."
            return self.default_vlm_output(message, "none", 0.0)

        if action_id is not None and action_id not in self.actions:
            message = "The proposed action is not from the allowed action list. I will not execute it."
            return self.default_vlm_output(message, "none", 0.0)

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        confidence = self.clamp(confidence, 0.0, 1.0)

        if match_type == "none":
            action_id = None

        if message is None or not isinstance(message, str):
            message = "I can’t generate a safe natural language message, so no action will run."

        result = {
            "intent": intent,
            "object": obj,
            "action_id": action_id,
            "match_type": match_type,
            "confidence": confidence,
            "message": message,
            "reasoning": reasoning,
        }

        return result

    def default_vlm_output(self, message, match_type, confidence):
        return {
            "intent": None,
            "object": None,
            "action_id": None,
            "match_type": match_type,
            "confidence": confidence,
            "message": message,
            "reasoning": None,
        }

    def apply_execution_policy(self, vlm_output):
        action_id = vlm_output.get("action_id")
        match_type = vlm_output.get("match_type")
        confidence = vlm_output.get("confidence", 0.0)

        if action_id is None:
            return False

        if match_type == "exact" and confidence >= self.min_exact_confidence:
            return True

        if match_type == "adapted" and confidence >= self.min_adapted_confidence:
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
        self.initialise_device()
        #self.connect_robot()
        self.process_frame() # Warmup

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
                    
                    self.process_frame()

                    if self.state == "idle":
                        self.encode_image()
                        vlm_output = self.query_vlm()

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
                                f"I propose action {vlm_output['action_id']} (match_type={vlm_output['match_type']}, "
                                f"confidence={vlm_output['confidence']:.2f}). Press Enter to confirm or ESC to cancel."
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
            if self.device:
                try:
                    self.device.close()
                except Exception:
                    pass

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
    parser.add_argument("--min_exact_confidence", type=float, default=0.75)
    parser.add_argument("--min_adapted_confidence", type=float, default=0.5)
    parser.add_argument("--no_tts", action="store_true")
    args = parser.parse_args()

    actions = DEFAULT_ACTIONS
    if args.actions_json:
        actions = load_actions_from_file(args.actions_json)

    assistant = GazeActionAssistant(
        ollama_host=args.ollama_host,
        actions=actions,
        model=args.model,
        min_exact_confidence=args.min_exact_confidence,
        min_adapted_confidence=args.min_adapted_confidence,
        no_tts=args.no_tts,
    )

    assistant.run()


if __name__ == "__main__":
    main()
