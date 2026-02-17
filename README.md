Overview: 

  This project implements a fully offline speech-to-text system integrated with a Retrieval-Augmented Generation (RAG) based action execution pipeline.

  The system captures real-time microphone input, detects speech boundaries using Voice Activity Detection (VAD), transcribes speech using Whisper (CPU optimized), retrieves the most relevant action using FAISS-based vector similarity search, and safely executes system-level commands.

  All models are stored locally inside the project’s `models/` directory. Internet access is required only during the first run to download model weights. After that, the system runs completely offline without relying on external cache directories such as HuggingFace or Torch Hub.

Build_Windows_Executable: 
  To create a standalone Windows executable that includes the locally stored models, use:

  pyinstaller --onefile --add-data "models;models" app\main.py

  The executable will be generated in:

  dist/

  The generated EXE runs without requiring a Python installation and includes all necessary model files for offline execution.

Model_Download_Instructions: 
  The following models are automatically downloaded on first run and stored inside the local `models/` directory:

  - Whisper tiny (via faster-whisper)
  - Silero VAD (loaded locally from silero_vad.jit)
  - SentenceTransformer all-MiniLM-L6-v2

  Internet is required only for the first run.

  After the models are downloaded, the system runs fully offline.

  Models are stored in:

  models/

Performance_Benchmarks: 
  Test Environment:

  Windows 11  
  CPU-only execution  
  Whisper tiny (int8 quantized)  
  Greedy decoding (beam_size = 1)

  Average Latency:

  For short commands (approximately 1–1.5 seconds of speech):

  STT: 400–700 ms  
  RAG Retrieval: < 20 ms  
  Execution: < 15 ms  
  Total: 450–750 ms  

  For longer utterances (around 2 seconds of speech):

  STT: ~900–1000 ms  
  Total: ~950–1100 ms  

  Latency increases proportionally with utterance duration since Whisper processes the entire detected speech segment. Retrieval and execution remain consistently fast.

System_Requirements: 
  Windows 10/11  
  Python 3.10+  
  Microphone  
  CPU (GPU not required)  

  No internet connection is required after the initial model download.

  Demo video: 

  A full system demonstration is available at the link below:
  https://drive.google.com/file/d/1O3Smnas8x87pAfEsTQvbQKyZn5RrVRCj/view?usp=sharing
