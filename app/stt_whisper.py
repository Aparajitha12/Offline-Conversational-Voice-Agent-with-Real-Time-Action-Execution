from faster_whisper import WhisperModel
import numpy as np
from .config import WHISPER_MODEL, WHISPER_COMPUTE_TYPE, SAMPLE_RATE

class WhisperSTT:
    def __init__(self):
        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE
        )
        self.model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def transcribe(self, audio):
        segments, info = self.model.transcribe(audio, beam_size=1, language="en")
        result = []
        for s in segments:
            result.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip()
            })
        return result
