import pyttsx3
import threading
import time

class TextToSpeech:
    def __init__(self):
        """Initialize the text-to-speech engine"""
        self.engine = None
        self.voice_thread = None
        self.is_speaking = False
        self.text_queue = []
        self.lock = threading.Lock()

    def _initialize_engine(self):
        """Initialize a new engine instance"""
        engine = pyttsx3.init()

        # Configure voice properties
        engine.setProperty('rate', 150)    # Speed of speech
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

        # Get available voices and set to a female voice if available
        voices = engine.getProperty('voices')
        if voices:
            for voice in voices:
                # Try to find a female voice
                if "female" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            # If no female voice found, use the first available voice
            if not engine.getProperty('voice') and voices:
                engine.setProperty('voice', voices[0].id)

        return engine

    def speak(self, text):
        """
        Speak the given text asynchronously

        Args:
            text (str): The text to be spoken
        """
        # Don't start a new speech if text is empty
        if not text or text.isspace():
            return

        with self.lock:
            # If not already speaking, start the speech thread
            if not self.is_speaking or not self.voice_thread or not self.voice_thread.is_alive():
                self.is_speaking = True
                self.text_queue = [text]
                self.voice_thread = threading.Thread(target=self._speak_worker)
                self.voice_thread.daemon = True
                self.voice_thread.start()
            else:
                # If already speaking, just update the text to be spoken next
                self.text_queue = [text]

    def _speak_worker(self):
        """Worker thread that handles speech requests"""
        try:
            while True:
                # Get the text to speak
                with self.lock:
                    if not self.text_queue:
                        self.is_speaking = False
                        break
                    text = self.text_queue.pop(0)

                # Create a new engine instance for each speech
                engine = self._initialize_engine()
                engine.say(text)
                engine.runAndWait()
                engine.stop()

                # Small delay to allow clean engine shutdown
                time.sleep(0.1)

        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            with self.lock:
                self.is_speaking = False

    def stop(self):
        """Stop any ongoing speech"""
        with self.lock:
            self.text_queue = []
            self.is_speaking = False
            # The engine will stop naturally when the thread sees empty queue

    def is_busy(self):
        """Check if the TTS engine is currently speaking"""
        with self.lock:
            return self.is_speaking