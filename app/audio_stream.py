import queue
import sounddevice as sd
import numpy as np
from .config import SAMPLE_RATE, CHANNELS, AUDIO_BLOCK_SEC

class MicStream:
    def __init__(self):
        self.q = queue.Queue()

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(indata.copy().reshape(-1))

    def __enter__(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * AUDIO_BLOCK_SEC),
            callback=self.callback
        )
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stream.stop()
        self.stream.close()

    def read(self):
        return self.q.get()
