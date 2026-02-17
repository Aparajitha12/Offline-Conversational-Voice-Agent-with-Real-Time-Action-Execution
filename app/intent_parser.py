import re
from dataclasses import dataclass

@dataclass
class ParsedCommand:
    intent: str
    confidence: float
    params: dict
    valid: bool
    error: str | None = None


def parse_command(text: str, rag_results, confidence_threshold=0.55) -> ParsedCommand:
    """
    Parses intent and extracts parameters from transcribed text.
    """

    if not rag_results:
        return ParsedCommand(None, 0.0, {}, False, "No intent retrieved")

    intent = rag_results[0]["intent"]
    confidence = rag_results[0]["score"]

    if confidence < confidence_threshold:
        return ParsedCommand(intent, confidence, {}, False, "Low confidence")

    params = {}
    text_lower = text.lower()

    # -----------------------------
    # Parameter Extraction Rules
    # -----------------------------

    # Web search
    if intent == "web_search":
        match = re.search(r"search web for (.+)", text_lower)
        if match:
            params["query"] = match.group(1)

    # Create file
    elif intent == "create_file":
        match = re.search(r"(?:named|called) (.+)", text_lower)
        if match:
            params["filename"] = match.group(1)

    # Delete file
    elif intent == "delete_file":
        match = re.search(r"(?:named|called) (.+)", text_lower)
        if match:
            params["filename"] = match.group(1)

    # List directory
    elif intent == "list_dir":
        match = re.search(r"in (.+)", text_lower)
        if match:
            params["directory"] = match.group(1)

    # Get time / simple commands need no params
    elif intent in ["open_calculator", "open_notepad", "open_browser", "get_time", "help"]:
        pass

    return ParsedCommand(intent, confidence, params, True)
