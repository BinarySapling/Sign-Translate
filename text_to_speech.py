import os
from gtts import gTTS
import pygame

class TextToSpeech:
    def __init__(self, cache_dir='speech_cache'):
        """
        Initialize Text-to-Speech handler with caching mechanism

        Args:
            cache_dir (str): Directory to store cached speech files
        """
        self.cache_dir = cache_dir
        self.current_audio = None

        # Initialize pygame mixer for audio playback
        pygame.mixer.init()

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, text):
        """
        Generate a unique filename for cached speech

        Args:
            text (str): Text to be converted to speech

        Returns:
            str: Path to cached audio file
        """
        # Create a filename by replacing spaces with underscores and adding .mp3
        filename = f"{text.replace(' ', '_')}.mp3"
        return os.path.join(self.cache_dir, filename)

    def speak(self, text):
        """
        Convert text to speech and play audio

        Args:
            text (str): Text to be spoken
        """
        if not text:
            return

        # Stop any currently playing audio
        pygame.mixer.music.stop()

        # Check if cached file exists, if not create it
        cache_path = self._get_cache_path(text)

        if not os.path.exists(cache_path):
            try:
                # Use Google Text-to-Speech to create MP3
                tts = gTTS(text=text, lang='en')
                tts.save(cache_path)
            except Exception as e:
                print(f"Text-to-Speech conversion error: {e}")
                return

        try:
            # Load and play the audio
            pygame.mixer.music.load(cache_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Audio playback error: {e}")

    def is_speaking(self):
        """
        Check if audio is currently playing

        Returns:
            bool: True if audio is playing, False otherwise
        """
        return pygame.mixer.music.get_busy()

    def stop(self):
        """
        Stop any currently playing audio
        """
        pygame.mixer.music.stop()