import os
import subprocess
from gtts import gTTS
from kokoro import KPipeline
import torch
import soundfile as sf
import pygame
from pynput import keyboard

class AudioManager:
    _mixer = None
    _waiting_track = None
    _booting_track = None
    _music_volume = 0.6
    
    @classmethod
    def get_mixer(cls):
        if cls._mixer is None:
            cls._mixer = pygame.mixer
            cls._mixer.init()
        return cls._mixer
    
    @classmethod
    def get_waiting_track(cls):
        if cls._waiting_track is None:
            cls._waiting_track = cls.get_mixer().Sound("media/waiting_music.mp3")
        return cls._waiting_track
    
    @classmethod
    def get_booting_track(cls):
        if cls._booting_track is None:
            cls._booting_track = cls.get_mixer().Sound("media/booting_up.mp3")
        return cls._booting_track

    @classmethod
    def set_music_volume(cls, volume: float):
        cls._music_volume = volume
        cls.get_mixer().set_volume(cls._music_volume)

    @classmethod
    def get_music_volume(cls):
        return cls._music_volume

def say(text, engine: str = "gtts"):
    if engine == "gtts":
        say_gtts(text)
    elif engine == "kokoro":
        say_kokoro(text)

def say_gtts(text):
    """
    Uses gTTS to generate speech and plays it.
    """
    temp_file = "/tmp/speech_output.mp3"    
    try:
        # Generate the audio file using gTTS and save it
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file)

        # Play the audio file
        say_text(temp_file)
        
    except Exception as e:
        # Log any other failure (like gTTS failing to contact Google)
        print(f"Failed to create or play audio file: {e}")
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def say_kokoro(text):
    """
    Uses Kokoro TTS to generate speech and plays it.
    """
    try:
        # Initialise the Kokoro TTS pipeline
        pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cuda')
        voice = 'af_bella'
        generator = pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')

        for i, (gs, ps, audio) in enumerate(generator):
            # Save the output to a file
            output_filename = f"test_output_{i}.wav"
            sf.write(output_filename, audio, 24000)
            say_text(output_filename)
    except Exception as e:
        print(f"Failed to create or play audio file: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(output_filename):
            os.remove(output_filename)

def say_text(file_path: str):
    """
    Plays a sound file using pygame.
    """
    try:
        mixer = AudioManager.get_mixer()
        # If anything was being played, stop it with a short fadeout
        mixer.fadeout(2000)
        mixer.music.load(file_path)
        mixer.set_volume(1.0)
        mixer.music.play()
        while mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Failed to play audio file: {e}")

def log_say(text: str, play_sounds: bool = True, play_engine: str = "gtts") -> None:
    """Logs the given text and optionally plays it as speech.

    Args:
        text (str): The text to log and speak.
        play_sounds (bool): Whether to play the speech audio.
        play_engine (str): The speech engine to use ('gtts' or 'kokoro').
    """
    emoji = "🤖 🔊" if play_sounds else "🤖"
    print(f"\n{'='*60}\n{emoji} {text}\n{'='*60}\n")
    if play_sounds:
        say(text, engine=play_engine)

def init_keyboard_listener():
    events = {
        "enter": False,
        "esc": False,
        "last_number": None,
    }

    def on_press(key):
        # Clear previous content to keep display neat
        print("\r\033[K", end="", flush=True)
        try:
            if hasattr(key, "char") and key.char in ["1", "2", "3", "4"]:
                events["last_number"] = int(key.char)

            elif key == keyboard.Key.right or key == keyboard.Key.enter:
                events["enter"] = True

            elif key == keyboard.Key.esc:
                events["esc"] = True
                return False

        except Exception as e:
            print(f"Keyboard listener error: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events

def play_waiting_music():
    AudioManager.get_waiting_track().play(-1) # Loop indefinitely
    AudioManager.set_music_volume(AudioManager.get_music_volume())

def stop_waiting_music():
    AudioManager.get_waiting_track().fadeout(2000)  # Fade out over 2 seconds

def play_booting_music():
    booting_track = AudioManager.get_booting_track()
    booting_track.set_volume(AudioManager.get_music_volume())
    booting_track.play(-1, fade_ms=4000)
    
def stop_booting_music():
    AudioManager.get_booting_track().fadeout(2000)  # Fade out over 2 seconds

def enable_audio_notifications():
    AudioManager.set_music_volume(0.6)

def disable_audio_notifications():
    AudioManager.set_music_volume(0.0)