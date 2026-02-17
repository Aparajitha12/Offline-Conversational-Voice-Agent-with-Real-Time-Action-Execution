import json
from datetime import datetime

def log_event(data):
    data["timestamp"] = datetime.now().isoformat()
    with open("assistant.log", "a") as f:
        f.write(json.dumps(data) + "\n")
