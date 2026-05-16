"""Персистентное хранилище чатов."""

import json
from pathlib import Path

CHATS_DIR = Path("/app/chats")
CHATS_DIR.mkdir(exist_ok=True)

chat_histories: dict[str, list[dict]] = {}
MAX_HISTORY = 40


def load_chat_from_disk(chat_id: str) -> list[dict]:
    chat_file = CHATS_DIR / f"{chat_id}.json"
    if chat_file.exists():
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_chat_to_disk(chat_id: str, messages: list[dict]):
    chat_file = CHATS_DIR / f"{chat_id}.json"
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def get_chat_history(chat_id: str) -> list[dict]:
    if chat_id not in chat_histories:
        chat_histories[chat_id] = load_chat_from_disk(chat_id)
    return chat_histories[chat_id]


def add_message_to_chat(chat_id: str, role: str, content: str):
    history = get_chat_history(chat_id)
    history.append({"role": role, "content": content})
    
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    
    chat_histories[chat_id] = history
    save_chat_to_disk(chat_id, history)


def get_all_chats() -> list[dict]:
    chats = []
    for chat_file in sorted(CHATS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        chat_id = chat_file.stem
        history = load_chat_from_disk(chat_id)
        title = "Пустой чат"
        for msg in history:
            if msg["role"] == "user":
                title = msg["content"][:50]
                if len(msg["content"]) > 50:
                    title += "..."
                break
        chats.append({
            "id": chat_id,
            "title": title,
            "message_count": len(history)
        })
    return chats


def delete_chat_from_disk(chat_id: str):
    chat_file = CHATS_DIR / f"{chat_id}.json"
    if chat_file.exists():
        chat_file.unlink()
    if chat_id in chat_histories:
        del chat_histories[chat_id]