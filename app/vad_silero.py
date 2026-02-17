import os
import sys
import torch
import numpy as np
from .config import SAMPLE_RATE, SILENCE_END_SEC


def get_models_path():
    """
    Returns correct models path.
    Works in both normal Python and PyInstaller .exe.
    """
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.getcwd()

    return os.path.join(base_path, "models")


class SileroVAD:
    def __init__(self):
        model_dir = get_models_path()
        model_path = os.path.join(model_dir, "silero_vad.jit")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}. "
                f"Please place silero_vad.jit inside the models/ folder."
            )

        # Load local JIT model
        self.model = torch.jit.load(model_path)
        self.model.eval()

        self.chunk_size = 512  # required for 16kHz
        self.buffer = np.array([], dtype=np.float32)

        self.reset()

    def reset(self):
        self.in_speech = False
        self.speech_buffer = []
        self.last_voice_time = 0.0
        self.time = 0.0

    def process(self, block):
        """
        block: float32 numpy array
        returns (audio, start_time, end_time) or None
        """

        # Append incoming audio
        self.buffer = np.concatenate((self.buffer, block))

        while len(self.buffer) >= self.chunk_size:
            chunk = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.chunk_size:]

            x = torch.from_numpy(chunk).unsqueeze(0)

            with torch.no_grad():
                speech_prob = self.model(x, SAMPLE_RATE).item()

            block_time = len(chunk) / SAMPLE_RATE
            self.time += block_time

            if speech_prob > 0.7:
                self.in_speech = True
                self.last_voice_time = self.time
                self.speech_buffer.append(chunk)
            else:
                if self.in_speech:
                    if (self.time - self.last_voice_time) > SILENCE_END_SEC:
                        utter = np.concatenate(self.speech_buffer)
                        end_time = self.last_voice_time
                        self.reset()
                        return utter, 0.0, end_time

        return None
