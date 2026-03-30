import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class GenerationRecord:
    user_id: int
    username: str
    original_prompt: str
    final_prompt: str
    variations: List[str]
    chosen_prompt: str
    prompt_model: str
    image_model: str
    image_count: int
    had_input_image: bool
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HistoryStore:
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or os.path.join("logs", "image_generation_history.jsonl")
        self._last_record_by_user: Dict[int, GenerationRecord] = {}
        self._ensure_log_dir()

    def save(self, record: GenerationRecord) -> None:
        self._last_record_by_user[record.user_id] = record
        with open(self.log_path, "a", encoding="utf-8") as output:
            output.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def get_last(self, user_id: int) -> Optional[GenerationRecord]:
        return self._last_record_by_user.get(user_id)

    def _ensure_log_dir(self) -> None:
        parent = os.path.dirname(self.log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
