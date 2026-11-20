import json
import os
from pathlib import Path
from typing import Any, Iterable


def _atomic_write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, data: Any) -> None:
    content = json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n"
    _atomic_write_text(path, content)


def atomic_write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, default=str)
        for record in records
    ]
    content = "\n".join(lines) + ("\n" if lines else "")
    _atomic_write_text(path, content)
