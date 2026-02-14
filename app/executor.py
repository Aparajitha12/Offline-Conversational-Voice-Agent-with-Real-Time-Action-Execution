import subprocess
import webbrowser
import os
from datetime import datetime

def execute(intent, params=None):
    if intent == "open_calculator":
        subprocess.Popen("calc.exe")
        return "Calculator opened."

    if intent == "open_notepad":
        subprocess.Popen("notepad.exe")
        return "Notepad opened."

    if intent == "open_browser":
        webbrowser.open("https://google.com")
        return "Browser opened."

    if intent == "get_time":
        return datetime.now().strftime("%H:%M:%S")

    if intent == "help":
        return "Available commands: calculator, notepad, browser, file ops."

    return "Intent not implemented."
