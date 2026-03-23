import os
import subprocess
from gtts import gTTS
from kokoro import KPipeline
import torch
import soundfile as sf
import pygame
from pynput import keyboard

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
        play_sound(temp_file)
        
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
            play_sound(output_filename)
    except Exception as e:
        print(f"Failed to create or play audio file: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(output_filename):
            os.remove(output_filename)

def play_sound(file_path: str):
    """
    Plays a sound file using pygame.
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
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
    print(f"\n{'='*60}\n🤖 {text}\n{'='*60}\n")
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
