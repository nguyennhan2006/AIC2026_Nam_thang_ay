from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable, Iterator


SCENE_RE = re.compile(r"^(?P<video>.+)_S(?P<number>\d+)$")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def scene_parts(scene_id: str) -> tuple[str, int]:
    match = SCENE_RE.match(scene_id)
    if not match:
        raise ValueError(f"scene_id does not follow <video_id>_S####: {scene_id}")
    return match.group("video"), int(match.group("number"))


def normalize_space(value: object) -> str:
    return " ".join(str(value or "").split())


def accent_fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")


def make_fts_query(text: str, match_all: bool = False) -> str:
    """Turn natural text into a safe FTS5 query.

    FTS5 operators from user text are intentionally discarded. ``OR`` gives
    retrieval-oriented recall; ``match_all=True`` is useful for strict filters.
    """

    tokens = re.findall(r"[^\W_]+", normalize_space(text).lower(), flags=re.UNICODE)
    unique_tokens = list(dict.fromkeys(tokens))
    if not unique_tokens:
        return ""
    operator = " AND " if match_all else " OR "
    return operator.join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique_tokens)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            target = (destination / info.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe path in {archive}: {info.filename}")
        zipped.extractall(destination)


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            result.append(path.resolve())
    return sorted(result)

