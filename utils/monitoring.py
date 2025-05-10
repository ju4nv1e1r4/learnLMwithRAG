import json
from datetime import datetime


class ConversationMonitor:
    def __init__(self, session_id: str, path_prefix: str = "data/monitoring"):
        self.session_id = session_id
        self.file_path = f"{path_prefix}/sessoin-{session_id}.json"
        self.sessions: list[dict[str, str]] = []

    def add_turn(self, user, llm):
        session = {
            "datetime": str(datetime.now()),
            "user": user,
            "assistant": llm,
        }
        self.sessions.append(session)

    def export_conversation(self):
        with open(self.file_path, "w") as f:
            json.dump(self.sessions, f, indent=4)
