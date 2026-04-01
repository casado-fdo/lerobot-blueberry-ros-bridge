import os
import subprocess
import logging
import threading
import random
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from gtts import gTTS
from kokoro import KPipeline
import torch
import soundfile as sf
import pygame
from pynput import keyboard


class IOManager:
    """
    Comprehensive I/O manager for handling audio, text logs, and keyboard inputs.
    """
    FAIL = 0
    SUCCESS = 1
    UPDATE = 2
    IDLE = 3
    
    def __init__(self, log_file: Optional[str] = None, audio_enabled: bool = True, tts_engine: str = "gtts"):
        self.audio_enabled = audio_enabled
        self.tts_engine = tts_engine
        self.log_file = log_file or f"logs/io_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._mixer = None
        self._processing_track = None
        self._booting_track = None
        self._logout_track = None
        self._music_volume = 0.6
        self._keyboard_listener = None
        self._keyboard_events = {
            "enter": False,
            "esc": False,
            "last_number": None,
            "assist": False,
            "cancel": False,
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False
        }
        self._event_callbacks: Dict[str, Callable] = {}
        self.audio_dir = "media/sound_effects"
        self.presets_dir = "media/tts_presets"
        if self.tts_engine == "kokoro":
            self._kokoro_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cuda')
            self._kokoro_voice = 'af_bella'
            list(self._kokoro_pipeline("Warming up.", voice=self._kokoro_voice, speed=1))  # consume the generator to actually warm up

        random.seed(datetime.now().timestamp())
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize audio if enabled
        if self.audio_enabled:
            self._init_audio()
    
        # Initialize keyboard listener
        self._start_keyboard_listener()

    def _setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("IOManager initialized")
    
    def _init_audio(self):
        """Initialize pygame mixer for audio playback."""
        try:
            self._mixer = pygame.mixer
            self._mixer.init()
            self.logger.info("Audio mixer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize audio mixer: {e}")
            self.audio_enabled = False
    
    def get_mixer(self):
        """Get pygame mixer instance."""
        if self._mixer is None and self.audio_enabled:
            self._init_audio()
        return self._mixer
    
    def get_processing_track(self):
        """Get processing music track."""
        if self._processing_track is None and self.audio_enabled:
            try:
                self._processing_track = self.get_mixer().Sound(f"{self.audio_dir}/processing.mp3")
            except Exception as e:
                self.logger.error(f"Failed to load processing track: {e}")
        return self._processing_track
    
    def get_booting_track(self):
        """Get booting music track."""
        if self._booting_track is None and self.audio_enabled:
            try:
                self._booting_track = self.get_mixer().Sound(f"{self.audio_dir}/booting_up.mp3")
            except Exception as e:
                self.logger.error(f"Failed to load booting track: {e}")
        return self._booting_track

    def get_logout_track(self):
        """Get logout music track."""
        if self._logout_track is None and self.audio_enabled:
            try:
                self._logout_track = self.get_mixer().Sound(f"{self.audio_dir}/booting_up.mp3")
            except Exception as e:
                self.logger.error(f"Failed to load logout track: {e}")
        return self._logout_track
    
    def set_music_volume(self, volume: float):
        """Set music volume (0.0 to 1.0)."""
        self._music_volume = max(0.0, min(1.0, volume))
        if self.audio_enabled and self.get_mixer():
            self.get_mixer().set_volume(self._music_volume)
        self.logger.info(f"Music volume set to {self._music_volume}")
    
    def get_music_volume(self) -> float:
        """Get current music volume."""
        return self._music_volume
    
    def say(self, text: str) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful, False otherwise
        """
        if not self.audio_enabled:
            self.logger.info(f"Audio disabled, would say: {text}")
            return False
            
        try:
            if self.tts_engine == "gtts":
                return self._say_gtts(text)
            elif self.tts_engine == "kokoro":
                return self._say_kokoro(text)
            else:
                self.logger.error(f"Unknown TTS engine: {self.tts_engine}")
                return False
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            return False
    
    def _say_gtts(self, text: str) -> bool:
        """Use gTTS for speech synthesis."""
        temp_file = "/tmp/speech_output.mp3"
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file)
            return self.play_audio_file(temp_file)
        except Exception as e:
            self.logger.error(f"gTTS synthesis failed: {e}")
            return False
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _say_kokoro(self, text: str) -> bool:
        """Use Kokoro TTS for speech synthesis or preset audio files."""
        # Check if text follows preset:code pattern
        if text.startswith("preset:"):
            preset_code = text[7:]  # Remove "preset:" prefix
            return self._play_preset_audio(preset_code)
        
        # Use regular Kokoro TTS for non-preset text
        try:
            generator = self._kokoro_pipeline(text, voice=self._kokoro_voice, speed=1, split_pattern=r'\n+')
            
            for i, (gs, ps, audio) in enumerate(generator):
                output_filename = f"/tmp/kokoro_output_{i}.wav"
                sf.write(output_filename, audio, 24000)
                self.play_audio_file(output_filename)
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            return True
        except Exception as e:
            self.logger.error(f"Kokoro synthesis failed: {e}")
            return False
    
    def _play_preset_audio(self, preset_code: str) -> bool:
        """Play a random audio file from the specified preset folder."""
        preset_dir = os.path.join(self.presets_dir, preset_code)
        
        if not os.path.exists(preset_dir):
            self.logger.error(f"Preset directory not found: {preset_dir}")
            return False
        
        # Get all audio files in the preset directory
        audio_files = [f for f in os.listdir(preset_dir) if f.endswith(('.wav', '.mp3'))]
        
        if not audio_files:
            self.logger.error(f"No audio files found in preset directory: {preset_dir}")
            return False
        
        # Select a random audio file
        selected_file = random.choice(audio_files)
        file_path = os.path.join(preset_dir, selected_file)
        
        self.logger.info(f"Playing preset audio: {preset_code} -> {selected_file}")
        return self.play_audio_file(file_path, volume=0.6)
    
    def play_audio_file(self, file_path: str, volume: float = 1.0) -> bool:
        """
        Play an audio file using pygame mixer.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.audio_enabled:
            return False
            
        try:
            mixer = self.get_mixer()
            mixer.stop()
            mixer.music.load(file_path)
            mixer.music.set_volume(volume)
            mixer.music.play()
            while mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except Exception as e:
            self.logger.error(f"Failed to play audio file {file_path}: {e}")
            return False

    def notify(self, type: int, message: str = None, speak: bool = True):
        """
        Notify the user with a sound and a message.
        
        Args:
            type: Type of notification (IOManager.SUCCESS, IOManager.FAIL, IOManager.UPDATE)
            message: Message to notify
        """
        if type == self.SUCCESS:
            self.play_audio_file(os.path.join(self.audio_dir, "success.mp3"))
        elif type == self.FAIL:
            self.play_audio_file(os.path.join(self.audio_dir, "fail.mp3"))
        elif type == self.UPDATE:
            self.play_audio_file(os.path.join(self.audio_dir, "update01.mp3"))
        elif type == self.IDLE:
            self.play_audio_file(os.path.join(self.audio_dir, "update0.mp3"))
        if message:
            self.log(message, level="info", speak=speak)
    
    def log(self, message: str, level: str = "info", speak: bool = False):
        """
        Log a message and optionally speak it.
        
        Args:
            message: Message to log
            level: Log level ("info", "warning", "error", "debug")
            speak: Whether to speak the message
        """
        # Log to file and console
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        
        # Print with formatting
        emoji = "🤖 🔊" if speak else "🤖"
        print(f"\n{'='*60}\n{emoji} {message}\n{'='*60}\n")
        
        # Speak if requested
        if speak:
            self.say(message)
    
    def _start_keyboard_listener(self):
        """Start keyboard listener for input events."""
        if self._keyboard_listener is not None:
            self.logger.warning("Keyboard listener already running")
            return
        
        def on_press(key):
            try:
                # Clear previous content
                print("\r\033[K", end="", flush=True)
                
                # Handle number keys
                if hasattr(key, "char") and key.char in ["1", "2", "3", "4"]:
                    self._keyboard_events["last_number"] = int(key.char)
                    self._trigger_event(f"number_{key.char}")
                
                # Handle special keys
                elif key == keyboard.Key.enter or key == keyboard.Key.space or key == keyboard.Key.right:
                    self._keyboard_events["enter"] = True
                    self._keyboard_events["assist"] = True
                    self._keyboard_events["exit_early"] = True
                    self._trigger_event("enter")
                    self._trigger_event("assist")
                    self._trigger_event("exit_early")
                
                elif key == keyboard.Key.left:
                    self._keyboard_events["rerecord_episode"] = True
                    self._keyboard_events["exit_early"] = True
                    self._trigger_event("rerecord_episode")
                    self._trigger_event("exit_early")
                
                elif key == keyboard.Key.esc:
                    self._keyboard_events["esc"] = True
                    self._keyboard_events["cancel"] = True
                    self._keyboard_events["stop_recording"] = True
                    self._keyboard_events["exit_early"] = True
                    self._trigger_event("esc")
                    self._trigger_event("cancel")
                    self._trigger_event("stop_recording")
                    self._trigger_event("exit_early")
                
                # Handle custom key bindings
                key_str = str(key).replace('Key.', '') if hasattr(key, 'name') else str(key)
                self._trigger_event(f"key_{key_str}")
                
            except Exception as e:
                self.logger.error(f"Keyboard listener error: {e}")
        
        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()
        self.logger.info("Keyboard listener started")
    
    def stop_keyboard_listener(self):
        """Stop keyboard listener."""
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
            self.logger.info("Keyboard listener stopped")
    
    def register_event_callback(self, event_name: str, callback: Callable):
        """
        Register a callback for a specific event.
        
        Args:
            event_name: Name of the event
            callback: Function to call when event occurs
        """
        self._event_callbacks[event_name] = callback
        self.logger.info(f"Registered callback for event: {event_name}")
    
    def _trigger_event(self, event_name: str):
        """Trigger an event and call its callback if registered."""
        if event_name in self._event_callbacks:
            try:
                self._event_callbacks[event_name]()
            except Exception as e:
                self.logger.error(f"Event callback failed for {event_name}: {e}")
    
    def get_keyboard_events(self) -> Dict[str, Any]:
        """Get current keyboard events state."""
        events = self._keyboard_events
        return events
    
    def reset_keyboard_events(self):
        """Reset keyboard events state."""
        self._keyboard_events["enter"] = False
        self._keyboard_events["esc"] = False
        self._keyboard_events["assist"] = False
        self._keyboard_events["cancel"] = False
        self._keyboard_events["last_number"] = None
        self._keyboard_events["exit_early"] = False
        self._keyboard_events["rerecord_episode"] = False
        self._keyboard_events["stop_recording"] = False
    
    def play_processing_music(self, loop: bool = True):
        """Play processing music."""
        if not self.audio_enabled:
            return
        
        track = self.get_processing_track()
        if track:
            track.play(-1 if loop else 1)
            track.set_volume(self._music_volume)
            self.logger.info("processing music started")
    
    def stop_processing_music(self):
        """Stop processing music."""
        if self._processing_track:
            self._processing_track.stop()
            self.logger.info("processing music stopped")
    
    def play_booting_music(self, loop: bool = True):
        """Play booting music."""
        if not self.audio_enabled:
            return    
        track = self.get_booting_track()
        if track:
            track.set_volume(self._music_volume)
            track.play(-1 if loop else 1, fade_ms=4000)
            self.logger.info("Booting music started")

    def play_logout_music(self): 
        """Play logout music."""
        if not self.audio_enabled:
            return    
        track = self.get_logout_track()
        if track:
            track.set_volume(self._music_volume)
            track.play(1, fade_ms=2000)
            self.logger.info("Logout sound played")
    
    def stop_logout_music(self):
        """Stop logout music with fadeout."""
        if self._logout_track:
            self._logout_track.fadeout(3000)
            self.logger.info("Logout music stopped")
    
    def stop_booting_music(self):
        """Stop booting music with fadeout."""
        if self._booting_track:
            self._booting_track.fadeout(2000)
            self.logger.info("Booting music stopped")
    
    def enable_audio_notifications(self):
        """Enable audio notifications."""
        self.audio_enabled = True
        self.set_music_volume(0.7)
        self.logger.info("Audio notifications enabled")
    
    def disable_audio_notifications(self):
        """Disable audio notifications."""
        self.audio_enabled = False
        self.set_music_volume(0.0)
        self.logger.info("Audio notifications disabled")
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_keyboard_listener()
        self.stop_processing_music()
        self.stop_booting_music()
        # Clean up Kokoro pipeline
        if self._kokoro_pipeline is not None:
            del self._kokoro_pipeline
            self._kokoro_pipeline = None
        if self._mixer:
            self._mixer.quit()
        self.logger.info("IOManager cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
