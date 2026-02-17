import os
from faster_whisper import WhisperModel


def get_models_path():
    """
    Returns correct models path.
    Works both in normal Python and PyInstaller .exe.
    """
    if getattr(__import__("sys"), 'frozen', False):
        # Running inside .exe
        base_path = __import__("sys")._MEIPASS
    else:
        base_path = os.getcwd()

    return os.path.join(base_path, "models")


class WhisperSTT:
    def __init__(self):
        model_dir = get_models_path()

        self.model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            download_root=model_dir
        )

    def transcribe(self, audio):
        """
        Returns list of segments with:
        - text
        - start time
        - end time
        """
        segments, _ = self.model.transcribe(audio)


        results = []

        for segment in segments:
            results.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end
            })

        return results
