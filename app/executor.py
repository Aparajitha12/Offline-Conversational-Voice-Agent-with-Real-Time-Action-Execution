import subprocess
import webbrowser
import os
from datetime import datetime


def execute(intent: str, params: dict | None = None) -> str:
    """
    Executes system action based on parsed intent.
    Returns a user-readable result string.
    """

    params = params or {}

    # ----------------------------
    # System Applications
    # ----------------------------

    if intent == "open_calculator":
        try:
            subprocess.Popen("calc.exe")
            return "Calculator opened."
        except Exception as e:
            return f"Failed to open calculator: {str(e)}"

    if intent == "open_notepad":
        try:
            subprocess.Popen("notepad.exe")
            return "Notepad opened."
        except Exception as e:
            return f"Failed to open notepad: {str(e)}"

    if intent == "open_browser":
        try:
            webbrowser.open("https://google.com")
            return "Browser opened."
        except Exception as e:
            return f"Failed to open browser: {str(e)}"

    # ----------------------------
    # Utility Commands
    # ----------------------------

    if intent == "get_time":
        return datetime.now().strftime("%H:%M:%S")

    if intent == "help":
        return (
            "Available commands: "
            "open calculator, open notepad, open browser, "
            "what time is it, help."
        )

    # ----------------------------
    # Unknown Intent
    # ----------------------------

    return "Intent not implemented."
