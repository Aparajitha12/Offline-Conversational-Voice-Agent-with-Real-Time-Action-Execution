import time

from .audio_stream import MicStream
from .vad_silero import SileroVAD
from .stt_whisper import WhisperSTT
from .rag_store import ActionRAG
from .executor import execute
from .logger import log_event
from .intent_parser import parse_command


def main():
    print("Initializing models...")
    stt = WhisperSTT()
    rag = ActionRAG()
    vad = SileroVAD()

    print("\nListening...")
    print("Waiting for speech...")

    speech_active = False

    with MicStream() as mic:
        try:
            while True:
                block = mic.read()
                result = vad.process(block)

                if result:
                    audio, start, end = result

                    if not speech_active:
                        print("\nSpeech detected - processing...")
                        speech_active = True

                    t0 = time.time()
                    segments = stt.transcribe(audio)
                    stt_time = time.time() - t0

                    text = " ".join([s["text"] for s in segments]).strip()

                    if not text:
                        speech_active = False
                        print("Waiting for speech...")
                        continue

                    print(f"[{start:.2f}s -> {end:.2f}s] {text}")

                    t1 = time.time()
                    rag_results = rag.retrieve(text)
                    rag_time = time.time() - t1

                    cmd = parse_command(text, rag_results)

                    if not cmd.valid:
                        print(f"Error: {cmd.error}")
                        speech_active = False
                        print("Waiting for speech...")
                        continue

                    t2 = time.time()
                    result_msg = execute(cmd.intent, cmd.params)
                    exec_time = time.time() - t2

                    print("Intent:", cmd.intent)
                    print("Confidence:", round(cmd.confidence, 3))
                    print("Result:", result_msg)

                    total = stt_time + rag_time + exec_time

                    print(
                        f"STT: {round(stt_time*1000)}ms | "
                        f"RAG: {round(rag_time*1000)}ms | "
                        f"EXEC: {round(exec_time*1000)}ms | "
                        f"TOTAL: {round(total*1000)}ms"
                    )

                    log_event({
                        "text": text,
                        "intent": cmd.intent,
                        "confidence": cmd.confidence,
                        "params": cmd.params,
                        "stt_ms": round(stt_time*1000),
                        "rag_ms": round(rag_time*1000),
                        "exec_ms": round(exec_time*1000),
                        "total_ms": round(total*1000)
                    })

                    speech_active = False
                    print("\nWaiting for speech...")

        except KeyboardInterrupt:
            print("\nStopping microphone...")
            mic.stop()
            mic.close()
            print("Exiting cleanly.")



if __name__ == "__main__":
    main()
