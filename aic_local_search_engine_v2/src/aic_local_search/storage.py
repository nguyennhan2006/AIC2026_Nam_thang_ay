from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .config import EngineConfig
from .records import KeyframeDocument, SceneDocument
from .utils import accent_fold, make_fts_query


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE scenes (
    scene_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    scene_no INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    clip_path TEXT NOT NULL DEFAULT '',
    representative_keyframe_id TEXT NOT NULL DEFAULT '',
    vector_row INTEGER NOT NULL UNIQUE,
    ocr_text TEXT NOT NULL DEFAULT '',
    transcript TEXT NOT NULL DEFAULT '',
    caption_vi TEXT NOT NULL DEFAULT '',
    caption_en TEXT NOT NULL DEFAULT '',
    speech_summary TEXT NOT NULL DEFAULT '',
    scene_type TEXT NOT NULL DEFAULT 'other',
    visible_text TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '',
    entities TEXT NOT NULL DEFAULT '',
    actions TEXT NOT NULL DEFAULT '',
    attributes TEXT NOT NULL DEFAULT '',
    relations TEXT NOT NULL DEFAULT '',
    event_text TEXT NOT NULL DEFAULT '',
    semantic_status TEXT NOT NULL DEFAULT 'missing',
    quality_status TEXT NOT NULL DEFAULT 'passed',
    quality_penalty REAL NOT NULL DEFAULT 1.0,
    quality_errors_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_scenes_video_time ON scenes(video_id, start_sec, end_sec);
CREATE INDEX idx_scenes_quality ON scenes(quality_status);

CREATE TABLE keyframes (
    keyframe_id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL REFERENCES scenes(scene_id),
    frame_idx INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    image_path TEXT NOT NULL,
    vector_row INTEGER NOT NULL UNIQUE,
    quality_score REAL NOT NULL DEFAULT 0,
    ocr_text TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_keyframes_scene_time ON keyframes(scene_id, timestamp_sec);

CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL REFERENCES scenes(scene_id),
    video_id TEXT NOT NULL,
    event_order INTEGER NOT NULL,
    relative_start_sec REAL NOT NULL,
    relative_end_sec REAL NOT NULL,
    absolute_start_sec REAL NOT NULL,
    absolute_end_sec REAL NOT NULL,
    description_vi TEXT NOT NULL DEFAULT '',
    description_en TEXT NOT NULL DEFAULT ''
);
CREATE INDEX idx_events_video_time ON events(video_id, absolute_start_sec, absolute_end_sec);

CREATE TABLE engine_meta (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);

CREATE VIRTUAL TABLE semantic_fts USING fts5(
    scene_id UNINDEXED, caption_vi, caption_en, event_text,
    tokenize='unicode61 remove_diacritics 2', prefix='2 3 4'
);
CREATE VIRTUAL TABLE ocr_fts USING fts5(
    scene_id UNINDEXED, ocr_text, visible_text,
    tokenize='unicode61 remove_diacritics 2', prefix='2 3 4'
);
CREATE VIRTUAL TABLE speech_fts USING fts5(
    scene_id UNINDEXED, transcript, speech_summary,
    tokenize='unicode61 remove_diacritics 2', prefix='2 3 4'
);
CREATE VIRTUAL TABLE tags_fts USING fts5(
    scene_id UNINDEXED, keywords, entities, actions, attributes, relations, scene_type,
    tokenize='unicode61 remove_diacritics 2', prefix='2 3 4'
);
CREATE VIRTUAL TABLE event_fts USING fts5(
    event_id UNINDEXED, scene_id UNINDEXED, description_vi, description_en,
    tokenize='unicode61 remove_diacritics 2', prefix='2 3 4'
);
"""


BRANCHES = {
    "semantic": ("semantic_fts", (0.0, 2.0, 1.5, 1.2)),
    "ocr": ("ocr_fts", (0.0, 2.0, 1.2)),
    "speech": ("speech_fts", (0.0, 2.0, 1.4)),
    "tags": ("tags_fts", (0.0, 1.5, 1.4, 1.4, 1.0, 1.0, 0.8)),
}


def _fts_text(value: str) -> str:
    """Store original and accent-folded text (notably Vietnamese ``đ``)."""

    folded = accent_fold(value)
    return value if folded == value else f"{value} {folded}"


def connect_database(path: Path, readonly: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) if readonly else sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -65536")
    return connection


def create_database(
    path: Path,
    scenes: Iterable[SceneDocument],
    keyframes: Iterable[KeyframeDocument],
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = connect_database(path)
    try:
        connection.executescript(SCHEMA_SQL)
        scene_list = list(scenes)
        connection.executemany(
            f"INSERT INTO scenes VALUES ({','.join('?' for _ in range(28))})",
            [
                (
                    scene.scene_id, scene.video_id, scene.scene_no,
                    scene.start_frame, scene.end_frame, scene.start_sec, scene.end_sec,
                    scene.clip_path, scene.representative_keyframe_id, scene.vector_row,
                    scene.ocr_text, scene.transcript, scene.caption_vi, scene.caption_en,
                    scene.speech_summary, scene.scene_type, scene.visible_text,
                    scene.keywords, scene.entities, scene.actions, scene.attributes,
                    scene.relations, scene.event_text, scene.semantic_status,
                    scene.quality_status, scene.quality_penalty,
                    json.dumps(scene.quality_errors, ensure_ascii=False),
                    json.dumps(scene.metadata, ensure_ascii=False),
                )
                for scene in scene_list
            ],
        )
        connection.executemany(
            "INSERT INTO semantic_fts VALUES (?,?,?,?)",
            [
                (s.scene_id, _fts_text(s.caption_vi), _fts_text(s.caption_en), _fts_text(s.event_text))
                for s in scene_list
            ],
        )
        connection.executemany(
            "INSERT INTO ocr_fts VALUES (?,?,?)",
            [(s.scene_id, _fts_text(s.ocr_text), _fts_text(s.visible_text)) for s in scene_list],
        )
        connection.executemany(
            "INSERT INTO speech_fts VALUES (?,?,?)",
            [
                (s.scene_id, _fts_text(s.transcript), _fts_text(s.speech_summary))
                for s in scene_list
            ],
        )
        connection.executemany(
            "INSERT INTO tags_fts VALUES (?,?,?,?,?,?,?)",
            [
                (
                    s.scene_id,
                    _fts_text(s.keywords),
                    _fts_text(s.entities),
                    _fts_text(s.actions),
                    _fts_text(s.attributes),
                    _fts_text(s.relations),
                    _fts_text(s.scene_type),
                )
                for s in scene_list
            ],
        )

        event_rows = []
        event_fts_rows = []
        for scene in scene_list:
            for index, event in enumerate(scene.temporal_events, 1):
                order = int(event.get("order", index))
                event_id = f"{scene.scene_id}_E{order:04d}"
                rel_start = float(event.get("start_sec", 0.0))
                rel_end = float(event.get("end_sec", rel_start))
                desc_vi = str(event.get("description_vi", ""))
                desc_en = str(event.get("description_en", ""))
                event_rows.append(
                    (
                        event_id, scene.scene_id, scene.video_id, order,
                        rel_start, rel_end, scene.start_sec + rel_start,
                        scene.start_sec + rel_end, desc_vi, desc_en,
                    )
                )
                event_fts_rows.append(
                    (event_id, scene.scene_id, _fts_text(desc_vi), _fts_text(desc_en))
                )
        connection.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", event_rows)
        connection.executemany("INSERT INTO event_fts VALUES (?,?,?,?)", event_fts_rows)

        scene_ids = {scene.scene_id for scene in scene_list}
        frame_rows = []
        for frame in keyframes:
            if frame.scene_id not in scene_ids:
                continue
            frame_rows.append(
                (
                    frame.keyframe_id, frame.scene_id, frame.frame_idx,
                    frame.timestamp_sec, frame.image_path, frame.vector_row,
                    frame.quality_score, frame.ocr_text,
                    json.dumps(frame.metadata, ensure_ascii=False),
                )
            )
        connection.executemany("INSERT INTO keyframes VALUES (?,?,?,?,?,?,?,?,?)", frame_rows)
        connection.executemany(
            "INSERT INTO engine_meta(key,value_json) VALUES (?,?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in meta.items()],
        )
        for table in (*[value[0] for value in BRANCHES.values()], "event_fts"):
            connection.execute(f"INSERT INTO {table}({table}) VALUES ('optimize')")
        connection.commit()
    finally:
        connection.close()


def _filter_sql(
    video_id: str | None,
    start_sec: float | None,
    end_sec: float | None,
    exclude_invalid: bool,
) -> tuple[list[str], list[object]]:
    where: list[str] = []
    params: list[object] = []
    if video_id:
        where.append("s.video_id = ?")
        params.append(video_id)
    if start_sec is not None:
        where.append("s.end_sec >= ?")
        params.append(float(start_sec))
    if end_sec is not None:
        where.append("s.start_sec <= ?")
        params.append(float(end_sec))
    if exclude_invalid:
        where.append("s.quality_status != 'invalid'")
    return where, params


def search_branch(
    connection: sqlite3.Connection,
    branch: str,
    text: str,
    limit: int,
    config: EngineConfig,
    video_id: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    match_all: bool = False,
) -> list[dict]:
    if branch not in BRANCHES:
        raise ValueError(f"Unknown lexical branch: {branch}")
    query = make_fts_query(text, match_all=match_all)
    if not query:
        return []
    table, weights = BRANCHES[branch]
    where, params = _filter_sql(video_id, start_sec, end_sec, config.exclude_invalid)
    where.insert(0, f"{table} MATCH ?")
    params.insert(0, query)
    placeholders = ",".join("?" for _ in weights)
    sql = f"""
        SELECT s.scene_id, bm25({table},{placeholders}) AS distance,
               snippet({table},-1,'[',']',' … ',28) AS snippet
        FROM {table} JOIN scenes s ON s.scene_id={table}.scene_id
        WHERE {' AND '.join(where)}
        ORDER BY distance ASC LIMIT ?
    """
    rows = connection.execute(sql, [*weights, *params, int(limit)]).fetchall()
    return [
        {
            "scene_id": row["scene_id"],
            "branch": branch,
            "score": -float(row["distance"]),
            "snippet": row["snippet"] or "",
        }
        for row in rows
    ]


def search_event_branch(
    connection: sqlite3.Connection,
    text: str,
    limit: int,
    config: EngineConfig,
    video_id: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> list[dict]:
    query = make_fts_query(text)
    if not query:
        return []
    where, params = _filter_sql(video_id, start_sec, end_sec, config.exclude_invalid)
    where.insert(0, "event_fts MATCH ?")
    params.insert(0, query)
    rows = connection.execute(
        f"""SELECT e.*, bm25(event_fts,0.0,0.0,2.0,1.5) AS distance,
                   snippet(event_fts,-1,'[',']',' … ',24) AS snippet
            FROM event_fts
            JOIN events e ON e.event_id=event_fts.event_id
            JOIN scenes s ON s.scene_id=e.scene_id
            WHERE {' AND '.join(where)}
            ORDER BY distance ASC LIMIT ?""",
        [*params, int(limit * 3)],
    ).fetchall()
    output: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if row["scene_id"] in seen:
            continue
        seen.add(row["scene_id"])
        output.append(
            {
                "scene_id": row["scene_id"],
                "branch": "event",
                "score": -float(row["distance"]),
                "snippet": row["snippet"] or "",
                "matched_event": {
                    key: row[key]
                    for key in (
                        "event_id", "event_order", "relative_start_sec", "relative_end_sec",
                        "absolute_start_sec", "absolute_end_sec", "description_vi", "description_en",
                    )
                },
            }
        )
        if len(output) >= limit:
            break
    return output


def _decode_scene(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["quality_errors"] = json.loads(item.pop("quality_errors_json"))
    item["metadata"] = json.loads(item.pop("metadata_json"))
    return item


def fetch_scenes(connection: sqlite3.Connection, scene_ids: list[str]) -> dict[str, dict]:
    if not scene_ids:
        return {}
    placeholders = ",".join("?" for _ in scene_ids)
    rows = connection.execute(f"SELECT * FROM scenes WHERE scene_id IN ({placeholders})", scene_ids).fetchall()
    return {row["scene_id"]: _decode_scene(row) for row in rows}


def fetch_scenes_by_vector_rows(connection: sqlite3.Connection, vector_rows: list[int]) -> dict[int, dict]:
    if not vector_rows:
        return {}
    placeholders = ",".join("?" for _ in vector_rows)
    rows = connection.execute(f"SELECT * FROM scenes WHERE vector_row IN ({placeholders})", vector_rows).fetchall()
    return {int(row["vector_row"]): _decode_scene(row) for row in rows}


def fetch_frames_by_vector_rows(connection: sqlite3.Connection, vector_rows: list[int]) -> dict[int, dict]:
    if not vector_rows:
        return {}
    placeholders = ",".join("?" for _ in vector_rows)
    rows = connection.execute(f"SELECT * FROM keyframes WHERE vector_row IN ({placeholders})", vector_rows).fetchall()
    output = {}
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        output[int(item["vector_row"])] = item
    return output


def representative_frame(connection: sqlite3.Connection, scene_id: str) -> dict | None:
    row = connection.execute(
        """SELECT k.* FROM keyframes k JOIN scenes s ON s.scene_id=k.scene_id
           WHERE k.scene_id=?
           ORDER BY (k.keyframe_id=s.representative_keyframe_id) DESC,
                    k.quality_score DESC, ABS(k.timestamp_sec-(s.start_sec+s.end_sec)/2.0) ASC
           LIMIT 1""",
        (scene_id,),
    ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["metadata"] = json.loads(item.pop("metadata_json"))
    return item
