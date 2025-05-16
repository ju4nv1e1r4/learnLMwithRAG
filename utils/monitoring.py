import json
import os
from datetime import datetime


class ConversationMonitor:
    def __init__(self, session_id: str, path_prefix: str = "data/monitoring"):
        self.session_id = session_id
        self.file_path = os.path.join(path_prefix, f"session-{session_id}.json")
        self.sessions: list[dict[str, str]] = []

    def add_turn(self, user, llm):
        session = {
            "datetime": str(datetime.now()),
            "user": str(user),
            "assistant": str(llm),
        }
        self.sessions.append(session)

    def export_conversation(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing conversation to {self.file_path}: {e}")


class ConversationLogProcessor:
    def format_sessions_to_text(self, sessions_data: list[dict[str, str]]) -> str:
        """
        Formats a list of session turns into a single string
        for semantic analysis. Accepts the sessions data directly.
        """
        if not sessions_data:
            return ""

        formatted_conversation = ""
        for turn in sessions_data:
            if "user" in turn and turn["user"]:
                formatted_conversation += f"User: {turn['user']}\n"
            if "assistant" in turn and turn["assistant"]:
                formatted_conversation += f"Assistant: {turn['assistant']}\n"

        return formatted_conversation.strip()

    def get_messages_from_sessions(
        self, sessions_data: list[dict[str, str]]
    ) -> list[str]:
        """
        Returns a list of messages (alternating user/assistant) from
        a list of session turns.
        """
        messages_list = []
        for turn in sessions_data:
            if "user" in turn and turn["user"]:
                messages_list.append(turn["user"])
            if "assistant" in turn and turn["assistant"]:
                messages_list.append(turn["assistant"])
        return messages_list


class ConversationLogManager:
    def __init__(self, session_id: str, path_prefix: str = "data/monitoring"):
        self.session_id = session_id
        self.monitor = ConversationMonitor(session_id, path_prefix)
        self.processor = ConversationLogProcessor()

    def add_turn(self, user: str, llm: str):
        self.monitor.add_turn(user, llm)

    def export_conversation(self):
        self.monitor.export_conversation()

    def get_formatted_text(self) -> str:
        return self.processor.format_sessions_to_text(self.monitor.sessions)

    def get_messages_list(self) -> list[str]:
        return self.processor.get_messages_from_sessions(self.monitor.sessions)
