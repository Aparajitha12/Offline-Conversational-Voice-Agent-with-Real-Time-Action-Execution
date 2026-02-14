import numpy as np
from .stt_whisper import WhisperSTT
from .rag_store import ActionRAG
from .executor import execute

def main():
    stt = WhisperSTT()
    rag = ActionRAG()

    print("Type something (simulate speech for now):")
    while True:
        text = input(">> ")

        if text.lower() == "exit":
            break

        rag_results = rag.retrieve(text)
        intent = rag_results[0]["intent"]

        result = execute(intent)
        print("Intent:", intent)
        print("Result:", result)

if __name__ == "__main__":
    main()
