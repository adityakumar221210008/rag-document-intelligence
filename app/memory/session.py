from collections import defaultdict, deque

MAX_HISTORY = 10  # turns per session


class SessionMemory:
    def __init__(self):
        self._store: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

    def add(self, session_id: str, user_msg: str, assistant_msg: str):
        self._store[session_id].append({
            "role": "user", "content": user_msg
        })
        self._store[session_id].append({
            "role": "assistant", "content": assistant_msg
        })

    def get(self, session_id: str) -> list[dict]:
        return list(self._store[session_id])

    def clear(self, session_id: str):
        self._store[session_id].clear()

    def list_sessions(self) -> list[str]:
        return list(self._store.keys())
