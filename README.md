## Quick Start Guide

To run the system locally, follow the steps below:

1. Clone the repository:

```bash
   git clone https://github.com/aparajitha12/offline-voice-assistant.git
   cd offline-voice-assistant
```

2. Create a virtual environment:

```bash
    python -m venv .venv
    .\.venv\Scripts\activate
```


  3. Install dependencies:

```bash
    pip install -r requirements.txt
```
    
  4. Run the application:

```bash
     python -m app.main
```

  On first execution, the required models will be downloaded and stored inside the local `models/` directory. After this initial setup, the system runs fully offline.

  ## Configuration Options

  System behavior can be adjusted through parameters defined in `app/config.py`.

  The most relevant configurable parameters include:

  - SAMPLE_RATE: Audio sampling rate used for microphone input.
  - SILENCE_END_SEC: Duration of silence required to mark the end of a speech segment. Lower values reduce latency but may cut speech early.
  - CONFIDENCE_THRESHOLD: Minimum similarity score required for an intent to be executed. Commands below this threshold are rejected for safety.
  - TOP_K: Number of candidate intents retrieved from FAISS during similarity search.
  - Whisper beam_size: Currently set to 1 for greedy decoding to minimize latency.

  These parameters allow balancing latency, responsiveness, and execution safety without modifying core logic.

  ## Architecture Overview

  The system follows a streaming architecture optimized for CPU-only offline execution.

  Speech-to-Text (STT):
  The system uses faster-whisper with the Whisper Tiny model, running on CPU with int8 quantization and beam_size set to 1. Whisper Tiny was selected because it provides a strong trade-off between accuracy and speed for short command-based interactions. Int8 quantization significantly reduces memory usage and inference time while maintaining acceptable transcription quality for command recognition.

  Voice Activity Detection (VAD):
  Silero VAD is used in streaming mode with fixed 512-sample chunks at 16kHz. This ensures speech detection operates in near real-time (32ms frames). By filtering out silence and background noise before transcription, unnecessary STT processing is avoided, directly reducing end-to-end latency.

  Retrieval-Augmented Action Execution (RAG):
  Transcribed text is converted into 384-dimensional embeddings using the all-MiniLM-L6-v2 SentenceTransformer model. These normalized embeddings are indexed using FAISS (IndexFlatIP) for cosine similarity search. Because the number of supported actions is small (10-50 intents), exact in-memory search is both efficient and sufficiently fast.

  Intent Parsing and Safety:
  Retrieved intents are validated using a confidence threshold. Commands below the threshold are rejected, preventing unintended execution. Only predefined intents from `data/actions.json` can be executed.

  Latency Optimization Strategy:
  Several optimizations were applied to achieve low latency:
  - Greedy decoding (beam_size = 1)
  - Int8 quantized Whisper model
  - 512-sample streaming VAD
  - Silence threshold tuning to reduce buffering delay
  - Precomputed and cached action embeddings
  - In-memory FAISS index

  The result is a system that typically achieves sub-second end-to-end latency for short commands while running entirely on CPU.


  ## Demo video: 

  A full system demonstration is available at the link below:
  https://drive.google.com/file/d/1O3Smnas8x87pAfEsTQvbQKyZn5RrVRCj/view?usp=sharing
