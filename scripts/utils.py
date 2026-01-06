import os
import subprocess
from gtts import gTTS

def say(text):
    """
    Replaces system-dependent 'say' with gTTS (Google Text-to-Speech)
    and plays it using the system's mpg123 command.
    """
    temp_file = "/tmp/speech_output.mp3"
    
    try:
        # 1. Create the audio file using gTTS and save it
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file)

        # 2. Play the audio file using the blocking 'subprocess.run'
        cmd = ["mpg123", temp_file]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    except Exception as e:
        # Log any other failure (like gTTS failing to contact Google)
        print(f"Failed to create or play audio file: {e}")
        
    finally:
        # 3. Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def log_say(text: str, play_sounds: bool = True) -> None:
    """Logs the given text and optionally plays it as speech.

    Args:
        text (str): The text to log and speak.
        play_sounds (bool): Whether to play the speech audio.
    """
    print(f"\n{'='*60}\n🤖 {text}\n{'='*60}\n")
    if play_sounds:
        say(text)