"""
AIC 2026 - Caption/Detect frame bang model Qwen3-VL (qua OpenRouter hoac vLLM tu host).

Doi tu notebook thu nghiem (notebooks/Route2 (1).py) sang scripts/ chinh thuc - logic
GIU NGUYEN 100% (da chay that qua OpenRouter, verify duoc: "THIENG LIENG" -> "THIÊNG
LIÊNG" dung, "Trong Hien" -> "Trọng Hiền" dung). Output cua script nay (JSONL scene-level,
schema "aic-multikeyframe-v2.0") duoc doc boi scripts/import_qwen3vl_captions.py de gop
vao Scene/Keyframe canonical (datasection) - xem file do de biet buoc tiep theo.

Ban chay script nay TU MAY LOCAL, no goi API toi server GPU ban vua thue (qua vLLM
OpenAI-compatible endpoint) thay vi goi OpenRouter. Logic giu nguyen tu notebook
(resume-able JSON, CoT observations, rule nhan manh dau tieng Viet cho OCR).

CACH DUNG:
    python -m scripts.caption_qwen3vl

Truoc khi chay, sua cac bien trong phan "USER CONFIG" ben duoi cho dung voi server
cua ban. Xem huong dan chi tiet o cuoi file.

LUU Y (theo quyet dinh cua team): model nay chi dung cho CAPTION (short/detailed,
entities, relations, scene_actions, keywords). OCR van giu nguyen o duong Qwen2.5-VL-7B
hien co trong offline/gpu_engine.py::_ocr_sync - ocr_regions sinh ra o day CHI de doi
chieu debug, KHONG duoc converter dua vao field OCR chinh thuc (tranh 2 nguon OCR da
nhau). Xem docs/14_TECHNICAL_PREPARATION.md muc "Da lam".
"""

from pathlib import Path
import os
import json
import time
import base64
import re
import mimetypes
import traceback
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import requests
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np

# =========================
# USER CONFIG - doc tu .env/bien moi truong, co fallback ve gia tri cu neu khong dat
# (giu nguyen hanh vi mac dinh cho ai dang chinh tay file nay nhu truoc) - xem
# .env.example muc "Qwen3-VL-32B caption (scripts/caption_qwen3vl.py)".
# =========================


def _read_env_file(env_path: Path, key_name: str) -> Optional[str]:
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key_name:
            return v.strip().strip('"').strip("'")
    return None


def _env(key_name: str) -> Optional[str]:
    # os.environ takes priority, but must distinguish "set to empty string" from "not
    # set at all" (a plain `or` chain treats both as falsy and would incorrectly fall
    # through to .env for an explicitly-empty shell/docker env var).
    if key_name in os.environ:
        return os.environ[key_name]
    return _read_env_file(Path(".env"), key_name)


def _env_int(key_name: str, default: Optional[int]) -> Optional[int]:
    """`key_name` absent -> `default`. Present but empty ("KEY=") -> None (explicit
    "no limit" - dùng cho AIC_QWEN3VL_LIMIT khi đã sẵn sàng chạy toàn bộ corpus)."""
    raw = _env(key_name)
    if raw is None:
        return default
    if not raw.strip():
        return None
    return int(raw)


PROVIDER_OPENROUTER = "openrouter"
PROVIDER_VLLM = "vllm"
VALID_PROVIDERS = {PROVIDER_OPENROUTER, PROVIDER_VLLM}


class ProviderConfig(NamedTuple):
    provider: str
    server_base_url: str
    model: str
    api_key: str


def resolve_provider_config(env: Callable[[str], Optional[str]]) -> ProviderConfig:
    """Resolve + validate the Qwen3-VL provider config from `env` (a key -> value
    lookup, normally `_env`). Pulled out as a pure function so the validation rules
    are unit-testable without touching real environment variables/.env.
    """

    provider = (env("AIC_QWEN3VL_PROVIDER") or PROVIDER_OPENROUTER).strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"AIC_QWEN3VL_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got {provider!r}"
        )

    if provider == PROVIDER_OPENROUTER:
        server_base_url = "https://openrouter.ai/api/v1"
        model = env("AIC_QWEN3VL_MODEL") or "qwen/qwen3-vl-32b-instruct"  # OpenRouter slug, lowercase
        api_key = env("OPENROUTER_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "AIC_QWEN3VL_PROVIDER=openrouter requires OPENROUTER_API_KEY trong .env "
                "hoac bien moi truong."
            )
        return ProviderConfig(provider, server_base_url, model, api_key)

    # provider == "vllm": server vLLM tu host cua ban (KHONG phai OpenRouter). Vi du neu
    # server co IP 123.45.67.89 va vLLM chay cong 8000:
    #   AIC_QWEN3VL_SERVER_URL=http://123.45.67.89:8000/v1
    # Neu thue tren Vast.ai, ho thuong cho 1 URL/port forward rieng - dung dung URL do.
    server_base_url = env("AIC_QWEN3VL_SERVER_URL") or ""
    if not server_base_url:
        raise ValueError(
            "AIC_QWEN3VL_PROVIDER=vllm requires AIC_QWEN3VL_SERVER_URL (OpenAI-compatible "
            "/v1 base URL, vd http://127.0.0.1:8001/v1)."
        )
    # Ten model PHAI KHOP CHINH XAC voi ten ban nap khi khoi dong vLLM server (vi du:
    # "Qwen/Qwen3-VL-32B-Instruct" hoac duong dan local ban dung khi chay lenh vllm serve).
    model = env("AIC_QWEN3VL_MODEL") or "Qwen/Qwen3-VL-32B-Instruct"
    # vLLM mac dinh KHONG yeu cau API key. Neu ban co bat --api-key khi chay vLLM server,
    # dien key do vao AIC_QWEN3VL_API_KEY. Neu khong, de nguyen chuoi bat ky, code se van
    # gui header nhung server se bo qua neu khong bat xac thuc.
    api_key = env("AIC_QWEN3VL_API_KEY") or "not-needed"
    return ProviderConfig(provider, server_base_url, model, api_key)


_provider_config = resolve_provider_config(_env)
USE_OPENROUTER = _provider_config.provider == PROVIDER_OPENROUTER  # dùng cho MAX_WORKERS/SLEEP mặc định bên dưới
SERVER_BASE_URL = _provider_config.server_base_url
MODEL = _provider_config.model
API_KEY = _provider_config.api_key

CHAT_COMPLETIONS_URL = f"{SERVER_BASE_URL}/chat/completions"

# Folder chua frame anh can caption. Du lieu that: da duoc chia san theo scene qua 1
# pipeline khac (offline/pipeline.py: scene uniform + keyframe select) - moi scene la 1
# subfolder "{video_id}_S{scene_idx}" duoi storage/processed/keyframes/{video_id}/, giu
# nguyen cau truc do. Doi lai cho dung voi du lieu that cua ban truoc khi chay.
FRAMES_DIR = Path("./storage/processed/keyframes")

# File metadata chinh xac di kem (scene_id/frame_idx/timestamp_sec cho tung keyframe_id),
# do pipeline tao keyframe sinh ra san - uu tien dung file nay thay vi tu doan qua regex
# ten file (regex van giu lam fallback cho dataset khac khong co file nay).
KEYFRAMES_METADATA_JSON = Path("./storage/exports/keyframes.jsonl")

# Output (JSONL - 1 JSON object/dong, append duoc truc tiep khong can doc lai ca file)
OUT_DIR = Path("./storage/exports/qwen3vl_captions")
OUT_JSON = OUT_DIR / "frame_captions_selfhosted.jsonl"
OUT_FAILED_JSON = OUT_DIR / "failed_frame_captions_selfhosted.jsonl"
OUT_INDEX_JSON = OUT_DIR / "frame_text_index_ready_selfhosted.jsonl"

# Bat CoT (observations truoc, giup giam hallucination, ton them token).
USE_COT = True

# Server tu host thi khong bi tinh tien theo token nhu API, nen co the tang thoai mai
# de tranh bi cat JSON giua chung (finish_reason = "length").
MAX_TOKENS = 1600 if USE_COT else 1200
TEMPERATURE = 0
TIMEOUT_SEC = 180
MAX_RETRIES = 4
# Goi qua OpenRouter can nghi giua cac request de tranh rate-limit; tu host thi khong can.
SLEEP_BETWEEN_CALLS = 0.2 if USE_OPENROUTER else 0.0

# Chay thu it anh/scene truoc khi chay full. Doi None (hoac AIC_QWEN3VL_LIMIT= rong) de
# chay toan bo. Luu y: o RUN_MODE="scene", LIMIT gioi han SO SCENE (khong phai so anh) - vd
# LIMIT=1 chi chay scene dau tien (theo thu tu sap xep ten scene_id).
LIMIT: Optional[int] = _env_int("AIC_QWEN3VL_LIMIT", 1)

# Tat resize: gui nguyen do phan giai goc de OCR chinh xac hon. Doi lai payload gui di
# se nang hon, ton nhieu prompt token/chi phi hon voi frame do phan giai cao.
RESIZE_BEFORE_SEND = False
MAX_SIDE = 1600
JPEG_QUALITY = 92
TMP_RESIZED_DIR = OUT_DIR / "_resized_tmp"

# So luong worker chay song song. Goi qua OpenRouter nen de thap (2-4) tranh rate-limit;
# khi chuyen sang server tu host co the tang len 4-16 tuy vRAM/GPU con trong.
# Mac dinh 2 cho ca 2 provider - Qwen3-VL-32B BF16 tren 1 A100 80GB rat sat VRAM (~66GB+KV,
# xem docs/05_VAST_DEPLOYMENT.md), tang dan 2->4->... sau khi do peak VRAM on dinh va
# throughput thuc tang, KHONG dat cao ngay tu dau khi chua benchmark.
MAX_WORKERS = _env_int("AIC_QWEN3VL_MAX_WORKERS", 2)

# So bin cho histogram HSV (16 la muc thong dung, du chi tiet ma khong qua nang).
HSV_HIST_BINS = 16

# Tu dong khoi phuc dau tieng Viet cho ocr_text sau khi caption xong (khong can chon tay
# tung anh). Model chi anh huong ocr_text - cac field khac (caption/keywords) da la tieng Anh.
# Can cai: pip install transformers torch huggingface_hub
ENABLE_ACCENT_RESTORE = True
ACCENT_MODEL_NAME = "peterhung/vietnamese-accent-marker-xlm-roberta"

# =========================
# CONFIG CHO CHE DO SCENE-LEVEL: gom nhieu frame cung 1 scene (nhom theo ten file/folder
# qua infer_video_id_from_path) gui chung 1 lan goi API, model tra ve metadata cho tung
# frame + 1 metadata tong hop cho ca scene.
# =========================

# "frame" = giu nguyen luong cu (moi anh 1 request).
# "scene" = gom nhom frame theo scene, gui nhieu anh/1 request.
RUN_MODE = "scene"

# Gioi han so frame gui trong 1 lan goi cho 1 scene (< 10 theo yeu cau — model/API de bi
# tu choi hoac giam chat luong neu nhoi qua nhieu anh vao 1 request).
MAX_FRAMES_PER_SCENE = 3

SCENE_SCHEMA_VERSION = "aic-multikeyframe-v2.0"

OUT_SCENE_JSON = OUT_DIR / "scene_captions_selfhosted.jsonl"
OUT_SCENE_FAILED_JSON = OUT_DIR / "failed_scene_captions_selfhosted.jsonl"
OUT_SCENE_INDEX_JSON = OUT_DIR / "scene_text_index_ready_selfhosted.jsonl"


# =========================
# PROMPT (giu nguyen tu notebook, da co rule nhan manh dau tieng Viet)
# =========================


def build_frame_metadata_prompt(use_cot: bool = True) -> str:
    cot_rule = ""
    cot_field = ""
    if use_cot:
        cot_rule = (
            '\n- Fill "observations" FIRST, before any other field. Use it to think step by step:\n'
            "  scan the whole image region by region, note every person/object/text/action you see,\n"
            "  then use only those noted observations to fill in the rest of the fields below.\n"
            '  Do not introduce anything in later fields that is not grounded in "observations".'
        )
        cot_field = (
            '  "observations": "step-by-step scratchpad: enumerate what is visible region by region '
            "(people, objects, text, actions, colors) before concluding. This field is internal reasoning, "
            'not shown to the end user.",\n'
        )

    return f"""
You are generating retrieval-oriented metadata for a video frame.

Return ONLY one valid JSON object. Do not use markdown. Do not explain outside the JSON.

Task:
Analyze the image carefully and produce structured metadata useful for video moment retrieval.

Rules:
- Describe only visible evidence in the image.
- Do not guess identities, names, brands, locations, dates, or text if not clearly visible.
- If visible text exists, copy it exactly into ocr_text.
- Vietnamese text almost always carries diacritics: tone marks (huyền, sắc, hỏi, ngã, nặng)
  and modified vowels (ă, â, ê, ô, ơ, ư). Read each character very carefully, especially on
  stylized, large, or 3D-rendered text. Do not drop diacritics — if a word looks like a
  Vietnamese word without accents, re-examine the image for missing tone/vowel marks before
  finalizing ocr_text.
- Any NON-PERSON object (a moving car, a running dog, a spinning fan, machinery in
  operation) performing a notable action must be included as an entry in main_subjects
  with its action and bbox_2d filled in. Do not leave action-bearing objects only as a
  plain name in the objects list. Do NOT put people in main_subjects — every person
  belongs only in the "people" list below, never duplicated in main_subjects.
- Use English keywords because they will be used for retrieval indexing.
- bbox_2d should be approximate normalized coordinates [x1, y1, x2, y2] in range 0-2000.
- If bbox is uncertain, use null.
- Keep captions factual and concise.
- Prefer common object/action words that help search.
- Do not mention that this is a frame or an image unless needed.{cot_rule}

JSON schema:
{{
{cot_field}  "short_caption": "one sentence summary",
  "detailed_caption": "2-4 factual sentences describing people, objects, actions, scene, background, colors, and visible text",
  "scene_type": "indoor/outdoor/street/room/screen/document/sports/news/vehicle/unknown",
  "main_subjects": [
    {{
      "name": "object/animal/vehicle/etc — NOT a person, people go in the people list only",
      "attributes": ["e.g. \\"red\\", \\"metallic\\", \\"open\\", \\"left side\\" — actual descriptive values, not category names"],
      "action": "visible action or null",
      "bbox_2d": [x1, y1, x2, y2]
    }}
  ],
  "people": [
    {{
      "description": "visible non-identifying description",
      "clothing": ["visible clothing"],
      "action": "visible action or null",
      "bbox_2d": [x1, y1, x2, y2]
    }}
  ],
  "objects": ["list of visible objects"],
  "actions": ["list of visible actions"],
  "attributes": ["important visual attributes: colors, weather, lighting, camera view, etc"],
  "relations": ["subject-object or spatial relations"],
  "ocr_text": [
    {{
      "text": "visible text exactly as seen",
      "bbox_2d": [x1, y1, x2, y2]
    }}
  ],
  "search_keywords": ["compact keywords for BM25/search"],
  "retrieval_notes": "one sentence explaining what queries this frame may match",
  "negative_uncertainty": ["important things that are unclear or not readable"]
}}
""".strip()


FRAME_METADATA_PROMPT = build_frame_metadata_prompt(USE_COT)


_SCENE_METADATA_PROMPT_TEMPLATE = r"""
You are generating retrieval-oriented metadata for a sequence of __NUM_FRAMES__ video
keyframes that all belong to the SAME continuous scene (consecutive in time). You will
receive __NUM_FRAMES__ images in order, each preceded by a text label "Frame 1:", "Frame 2:",
etc. Use these EXACT labels ("1", "2", ...) as keyframe_id / supporting_keyframe_ids values.

Return ONLY one valid JSON object matching schema_version "__SCHEMA_VERSION__" below.
Do not use markdown. Do not explain outside the JSON.

Task:
1. Fill "scene_context": one aggregated description of the whole scene, including entities
   that recur across keyframes (scene_entities), notable actions across the scene
   (scene_actions — mark motion_verified=true and evidence_type="multi_keyframe_inference"
   ONLY if you can compare motion/position across 2+ keyframes; otherwise use
   evidence_type="single_keyframe_inference" and motion_verified=false), and which single
   keyframe best represents the scene (best_keyframe_id).
2. Fill "keyframes": one entry per image you were given, in the exact same order, each with
   its own entities/relations/ocr_regions grounded ONLY in that specific image.
   (If only 1 image is given, there is nothing to compare motion against — every
   scene_action must use evidence_type="single_keyframe_inference" and motion_verified=false.)

General rules:
- Describe only visible evidence. Do not guess identities, names, brands, locations, or dates.
- Every caption/description/keyword field has an "_en" (English) and "_vi" (Vietnamese)
  version — both must be filled, as accurate translations of the same factual content.
- Exception: "ocr_regions[].text_raw" is copied EXACTLY as seen in the image, in whatever
  language it actually appears in — never translate OCR text, only transcribe it.
- Vietnamese text in the image almost always carries diacritics: tone marks (huyền, sắc,
  hỏi, ngã, nặng) and modified vowels (ă, â, ê, ô, ơ, ư). Read very carefully, especially on
  stylized/large/3D text. If a transcribed word looks like a Vietnamese word missing its
  accents, re-examine before finalizing text_raw, and mark the span in "uncertain_spans" if
  still unsure rather than guessing.
- For every ocr_regions entry, set "difficulty_flags" honestly and set
  "model_retry_recommended": true whenever readability is "partial"/"unreadable" or any
  difficulty_flag is true and the text seems important (a name, a place, a number) — this
  flag is used downstream to decide which frames get a zoomed-in re-check.
- bbox_2d is normalized [x1, y1, x2, y2] in range 0-2000. Use null if uncertain.
- scene_entity_id / entity_id / scene_action_id / relation_id / evidence_id / region_id must
  be short stable ids (e.g. "SE001", "E001", "SA001", "R001", "VE001", "OCR001"), unique
  within their list. When a keyframe entity is the same real-world thing as a scene_entity,
  set the keyframe entity's "scene_entity_id" to that same id — leave it null if you are not
  confident they are the same.
- Any object (not just people) performing a notable action must appear as an entity with its
  "action" filled in, not just listed as a bare name.
- Every entity has its own "certainty" (certain|likely|possible|unclear) about whether that
  entity is correctly identified at all (occlusion, bad angle, ambiguous shape) — this is
  separate from "action.certainty" (which is about the ACTION, not the entity's identity).
  Every attribute also has its own "certainty" (e.g. a color guessed under poor lighting is
  "possible", not "certain"). Only add a matching entry to "uncertainties" when you need to
  explain WHY in free text — the certainty field itself must always be filled, do not rely on
  "uncertainties" alone to signal doubt.
- Keep every caption factual and concise. English keywords/descriptions favor common
  object/action words that help search.
- Do not mention that this is a frame/keyframe/image unless the content itself requires it.

JSON schema (fill every field; use null/[] where genuinely nothing applies):
__JSON_SCHEMA__
""".strip()


_SCENE_JSON_SCHEMA_BODY = r"""{
  "schema_version": "__SCHEMA_VERSION__",
  "scene_id": "string",
  "scene_context": {
    "environment": "indoor|outdoor|mixed|unknown",
    "setting": "street|room|water_body|sports_field|stage|vehicle_interior|document|screen|nature|other|unknown",
    "media_type": "real_world|news|sports_broadcast|document|screen_capture|animation|unknown",
    "short_caption_en": "string",
    "short_caption_vi": "string",
    "detailed_caption_en": "string",
    "detailed_caption_vi": "string",
    "scene_entities": [
      {
        "scene_entity_id": "SE001",
        "type": "person|group|animal|vehicle|object|structure|place|natural_element|plant|food|text_region|screen_content|unknown",
        "name_en": "string",
        "name_vi": "string",
        "subtype": "string|null",
        "description_en": "string",
        "description_vi": "string",
        "supporting_keyframe_ids": ["string"]
      }
    ],
    "visual_evidence": [
      {
        "evidence_id": "VE001",
        "keyframe_ids": ["string"],
        "observation_en": "string",
        "observation_vi": "string",
        "supports": ["scene_caption|scene_action|entity_attribute|relation|ocr"]
      }
    ],
    "scene_actions": [
      {
        "scene_action_id": "SA001",
        "subject_scene_entity_id": "string|null",
        "label_en": "string",
        "label_vi": "string",
        "aliases_en": ["string"],
        "aliases_vi": ["string"],
        "object_scene_entity_id": "string|null",
        "object_text_en": "string|null",
        "object_text_vi": "string|null",
        "certainty": "certain|likely|possible|unclear",
        "evidence_type": "single_keyframe_inference|multi_keyframe_inference",
        "motion_verified": false,
        "supporting_keyframe_ids": ["string"],
        "best_keyframe_id": "string|null"
      }
    ],
    "best_keyframe_id": "string|null",
    "best_keyframe_reason_en": "string|null",
    "best_keyframe_reason_vi": "string|null",
    "uncertainties": [
      {
        "scope": "scene|scene_entity|scene_action|relation|ocr",
        "ref_id": "string|null",
        "description_en": "string",
        "description_vi": "string"
      }
    ]
  },
  "keyframes": [
    {
      "keyframe_id": "string",
      "short_caption_en": "string",
      "short_caption_vi": "string",
      "detailed_caption_en": "string",
      "detailed_caption_vi": "string",
      "entities": [
        {
          "entity_id": "E001",
          "scene_entity_id": "string|null",
          "type": "person|group|animal|vehicle|object|structure|place|natural_element|plant|food|text_region|screen_content|unknown",
          "certainty": "certain|likely|possible|unclear",
          "name_en": "string",
          "name_vi": "string",
          "subtype": "string|null",
          "description_en": "string",
          "description_vi": "string",
          "attributes": [
            {
              "category": "color|clothing|material|pattern|shape|size|pose|state|species|vehicle_type|object_subtype|structure_type|place_type|natural_element_type|content_type|other",
              "name_en": "string",
              "name_vi": "string",
              "value_en": "string|null",
              "value_vi": "string|null",
              "body_part": "head|upper_body|lower_body|feet|full_body|null",
              "certainty": "certain|likely|possible|unclear"
            }
          ],
          "spatial": {
            "depth": "foreground|middle_ground|background|unknown",
            "visibility": "fully_visible|partially_visible|occluded|truncated|unknown"
          },
          "action": {
            "scene_action_id": "string|null",
            "label_en": "string|null",
            "label_vi": "string|null",
            "aliases_en": ["string"],
            "aliases_vi": ["string"],
            "certainty": "certain|likely|possible|unclear|null",
            "evidence_type": "directly_visible|visible_pose|single_keyframe_inference|scene_multi_keyframe_inference|unclear|null",
            "present_in_keyframe": "true|false|unclear",
            "motion_verified": false,
            "supporting_keyframe_ids": ["string"]
          },
          "bbox_2d": [x1, y1, x2, y2]
        }
      ],
      "relations": [
        {
          "relation_id": "R001",
          "subject_entity_id": "string",
          "predicate_normalized": "holding|wearing|inside|on|under|above|behind|in_front_of|next_to|looking_at|interacting_with|riding|carrying|using|touching|attached_to|entering|leaving|facing|other",
          "predicate_en": "string",
          "predicate_vi": "string",
          "object_entity_id": "string|null",
          "object_text_en": "string|null",
          "object_text_vi": "string|null",
          "certainty": "certain|likely|possible|unclear"
        }
      ],
      "ocr_regions": [
        {
          "region_id": "OCR001",
          "bbox_2d": [x1, y1, x2, y2],
          "text_raw": "string",
          "language": "vi|en|mixed|unknown",
          "text_type": "subtitle|sign|overlay|document|screen|logo|label|other|unknown",
          "readability": "clear|partial|unreadable",
          "uncertain_spans": ["string"],
          "difficulty_flags": {
            "small_text": false,
            "blurred": false,
            "low_contrast": false,
            "stylized": false,
            "rotated": false,
            "curved": false,
            "occluded": false,
            "truncated": false
          },
          "model_retry_recommended": false
        }
      ],
      "keywords_en": ["string"],
      "keywords_vi": ["string"],
      "frame_selection": {
        "roles": ["representative|best_action_evidence|best_object_evidence|best_attribute_evidence|best_relation_evidence|best_ocr_evidence|context_only"],
        "evidence_strength": "strong|moderate|weak",
        "is_scene_best": false
      },
      "uncertainties": [
        {
          "scope": "keyframe|entity|action|relation|ocr|bbox",
          "ref_id": "string|null",
          "description_en": "string",
          "description_vi": "string"
        }
      ]
    }
    // ... one object per keyframe, __NUM_FRAMES__ total, in the same order as the images given,
    // with keyframe_id equal to "1", "2", ... matching the Frame N: labels given
  ]
}"""


def build_scene_metadata_prompt(use_cot: bool, num_frames: int) -> str:
    """
    Prompt cho che do scene-level, dung schema day du "aic-multikeyframe-v2.0" (song ngu
    EN/VI, entity/action/relation/ocr co cau truc sau, cross-reference id giua scene_context
    va keyframes). use_cot khong dieu khien them field rieng o day - do sau CoT da nam san
    trong "observations"-style fields (vd best_keyframe_reason, uncertainties) cua schema.
    """
    keyframe_list = ", ".join(str(i + 1) for i in range(num_frames))
    schema_body = _SCENE_JSON_SCHEMA_BODY.replace("__SCHEMA_VERSION__", SCENE_SCHEMA_VERSION)
    schema_body = schema_body.replace("__NUM_FRAMES__", str(num_frames))

    prompt = _SCENE_METADATA_PROMPT_TEMPLATE.replace("__NUM_FRAMES__", str(num_frames))
    prompt = prompt.replace("__SCHEMA_VERSION__", SCENE_SCHEMA_VERSION)
    prompt = prompt.replace("__JSON_SCHEMA__", schema_body)
    prompt += f"\n\n(keyframe_id values to use, in order: {keyframe_list})"
    return prompt


# =========================
# HELPERS (giu nguyen logic tu notebook)
# =========================

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def list_images(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists():
        print(f"WARNING: frames_dir does not exist: {frames_dir}")
        return []
    return sorted(
        [
            p
            for p in frames_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
    )


def infer_video_id_from_path(path: Path) -> str:
    parent = path.parent.name
    if parent and parent not in {"frames", "images", "keyframes"}:
        return parent
    stem = path.stem
    m = re.match(r"(.+?)_(?:frame|f)[_-]?\d+", stem)
    return m.group(1) if m else parent


def infer_frame_idx(path: Path) -> Optional[int]:
    stem = path.stem
    patterns = [
        r"frame[_-]?(\d+)",
        r"_f(\d+)",
        r"(?:^|_)(\d{4,})(?:_|$)",
        r"(\d+)$",
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def infer_timestamp_from_name(path: Path) -> Optional[float]:
    stem = path.stem
    m = re.search(r"_t(\d+(?:\.\d+)?)", stem)
    if m:
        return float(m.group(1))
    return None


# =========================
# METADATA THAT TU keyframes.json (uu tien hon regex doan mo o tren, neu file ton tai)
# =========================

_keyframes_meta_by_stem: Optional[Dict[str, Dict[str, Any]]] = None


def _load_keyframes_metadata() -> Dict[str, Dict[str, Any]]:
    """Doc KEYFRAMES_METADATA_JSON 1 lan (lazy), lap chi muc theo keyframe_id (= ten file
    khong duoi, vi du 'K16_V001_S0001_F000000'). Neu file khong ton tai, tra ve dict rong
    va moi getter ben duoi se tu dong fallback ve ham regex infer_* nhu cu.

    Ho tro ca 2 dinh dang: JSON array (dinh dang cu cua notebook) va JSONL (dinh dang that
    cua storage/exports/keyframes.jsonl do datasection/exporter.py sinh ra - moi dong la
    1 object Keyframe).
    """
    global _keyframes_meta_by_stem
    if _keyframes_meta_by_stem is not None:
        return _keyframes_meta_by_stem

    _keyframes_meta_by_stem = {}
    if KEYFRAMES_METADATA_JSON.exists():
        try:
            text = KEYFRAMES_METADATA_JSON.read_text(encoding="utf-8")
            stripped = text.lstrip()
            if stripped.startswith("["):
                data = json.loads(text)
            else:
                data = [json.loads(line) for line in text.splitlines() if line.strip()]
            for item in data:
                kf_id = item.get("keyframe_id")
                if kf_id:
                    _keyframes_meta_by_stem[kf_id] = item
            print(f"Da nap {len(_keyframes_meta_by_stem)} keyframe metadata tu {KEYFRAMES_METADATA_JSON}")
        except Exception as e:
            print(f"WARNING: khong doc duoc {KEYFRAMES_METADATA_JSON}: {repr(e)}")
    return _keyframes_meta_by_stem


def get_scene_id_for_path(path: Path) -> str:
    meta = _load_keyframes_metadata().get(path.stem)
    if meta and meta.get("scene_id"):
        return meta["scene_id"]
    return infer_video_id_from_path(path)


def get_frame_idx_for_path(path: Path) -> Optional[int]:
    meta = _load_keyframes_metadata().get(path.stem)
    if meta and meta.get("frame_idx") is not None:
        return meta["frame_idx"]
    return infer_frame_idx(path)


def get_timestamp_for_path(path: Path) -> Optional[float]:
    meta = _load_keyframes_metadata().get(path.stem)
    if meta and meta.get("timestamp_sec") is not None:
        return meta["timestamp_sec"]
    return infer_timestamp_from_name(path)


def group_frames_by_scene(images: List[Path]) -> Dict[str, List[Path]]:
    """
    Nhom frame theo scene. Uu tien scene_id that tu keyframes.json (get_scene_id_for_path);
    neu khong co file do, tu dong fallback ve infer_video_id_from_path() (nhom theo ten
    folder con hoac prefix ten file).
    """
    groups: Dict[str, List[Path]] = {}
    for p in images:
        key = get_scene_id_for_path(p)
        groups.setdefault(key, []).append(p)
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


def sample_frames_even(paths: List[Path], max_count: int) -> List[Path]:
    """Neu scene co qua nhieu frame, lay mau deu (dau/giua/cuoi xen ke) thay vi lay het."""
    if len(paths) <= max_count:
        return paths
    step = len(paths) / max_count
    idxs = sorted({min(int(round(i * step)), len(paths) - 1) for i in range(max_count)})
    i = 0
    while len(idxs) < max_count and i < len(paths):
        if i not in idxs:
            idxs.append(i)
            idxs = sorted(set(idxs))
        i += 1
    return [paths[i] for i in idxs[:max_count]]


def resize_image_for_api(image_path: Path, max_side: int = 1280) -> Path:
    if not RESIZE_BEFORE_SEND:
        return image_path

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale >= 1.0:
            return image_path

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        out_path = TMP_RESIZED_DIR / f"{image_path.stem}_max{max_side}.jpg"
        img.save(out_path, quality=JPEG_QUALITY)
        return out_path


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        suffix = image_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        else:
            mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json_from_text(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def flatten_for_index(parsed: Dict[str, Any]) -> str:
    if not parsed:
        return ""

    parts = []
    for key in ["short_caption", "detailed_caption", "scene_type", "retrieval_notes"]:
        val = parsed.get(key)
        if isinstance(val, str):
            parts.append(val)

    for key in [
        "objects",
        "actions",
        "attributes",
        "relations",
        "search_keywords",
        "negative_uncertainty",
    ]:
        val = parsed.get(key)
        if isinstance(val, list):
            parts.extend([str(x) for x in val if x is not None])

    for item in parsed.get("main_subjects", []) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("name", "")))
            parts.append(str(item.get("action", "")))
            attrs = item.get("attributes") or []
            if isinstance(attrs, list):
                parts.extend(map(str, attrs))

    for item in parsed.get("people", []) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("description", "")))
            parts.append(str(item.get("action", "")))
            clothing = item.get("clothing") or []
            if isinstance(clothing, list):
                parts.extend(map(str, clothing))

    for item in parsed.get("ocr_text", []) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("text", "")))
        elif isinstance(item, str):
            parts.append(item)

    return " ".join([p.strip() for p in parts if str(p).strip()])


# =========================
# TRICH XUAT HSV (chay local bang OpenCV, khong goi API, khong ton token)
# =========================


def extract_hsv_features(image_path: Path, bins: int = HSV_HIST_BINS) -> Dict[str, Any]:
    """
    Tinh histogram Hue/Saturation/Value + do sang trung binh cho 1 frame.
    Dung cho tim kiem/loc theo mau sac sau nay (vi du query "canh mau do", "canh toi").
    Chay hoan toan local (OpenCV), khong goi API nen khong ton chi phi.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten() / total_pixels
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten() / total_pixels
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten() / total_pixels

    dominant_hue_bin = int(np.argmax(hist_h))
    dominant_hue_deg = round(dominant_hue_bin * (180.0 / bins), 1)

    return {
        "hue_hist": [round(float(x), 4) for x in hist_h],
        "sat_hist": [round(float(x), 4) for x in hist_s],
        "val_hist": [round(float(x), 4) for x in hist_v],
        "dominant_hue_deg": dominant_hue_deg,
        "mean_saturation": round(float(np.mean(hsv[:, :, 1])), 2),
        "mean_brightness": round(float(np.mean(hsv[:, :, 2])), 2),
    }


# =========================
# KHOI PHUC DAU TIENG VIET (post-processing tu dong cho ocr_text, chay local)
# Model: peterhung/vietnamese-accent-marker-xlm-roberta (token classification).
# Da verify: "THIENG LIENG" -> "THIÊNG LIÊNG" dung, "Trong Hien" -> "Trọng Hiền" dung.
# Han che da biet: ten rieng ngan, dung 1 minh khong co ngu canh co the bi doan sai.
# =========================

_accent_model = None
_accent_tokenizer = None
_accent_label_list = None
_accent_device = None
_accent_load_failed = False
_ACCENT_WORD_PREFIX = "▁"  # "▁"


def _load_accent_model():
    global _accent_model, _accent_tokenizer, _accent_label_list, _accent_device, _accent_load_failed
    if _accent_model is not None or _accent_load_failed:
        return

    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        import torch

        tags_path = hf_hub_download(
            repo_id=ACCENT_MODEL_NAME, filename="selected_tags_names.txt"
        )
        with open(tags_path, "r", encoding="utf-8") as f:
            _accent_label_list = [line.strip() for line in f if line.strip()]

        _accent_tokenizer = AutoTokenizer.from_pretrained(
            ACCENT_MODEL_NAME, add_prefix_space=True
        )
        _accent_model = AutoModelForTokenClassification.from_pretrained(
            ACCENT_MODEL_NAME
        )
        _accent_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _accent_model.to(_accent_device)
        _accent_model.eval()
    except Exception as e:
        print(
            f"WARNING: khong load duoc accent restore model, bo qua buoc nay. Loi: {repr(e)}"
        )
        _accent_load_failed = True


def _accent_predict(text: str):
    import torch
    import numpy as np_

    tokens_in = text.strip().split()
    if not tokens_in:
        return [], []
    inputs = _accent_tokenizer(
        tokens_in,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    tokens = _accent_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]

    with torch.no_grad():
        inputs = {k: v.to(_accent_device) for k, v in inputs.items()}
        outputs = _accent_model(**inputs)

    predictions = np_.argmax(outputs["logits"].cpu().numpy(), axis=2)[0][1:-1]
    return tokens, predictions


def _accent_merge(tokens, predictions):
    merged = []
    i = 0
    while i < len(tokens):
        if tokens[i].startswith(_ACCENT_WORD_PREFIX):
            word_parts = [tokens[i][len(_ACCENT_WORD_PREFIX) :]]
            label_set = {predictions[i]}
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith(_ACCENT_WORD_PREFIX):
                word_parts.append(tokens[j])
                label_set.add(predictions[j])
                j += 1
            merged.append(("".join(word_parts), label_set))
            i = j
        else:
            merged.append((tokens[i], {predictions[i]}))
            i += 1
    return merged


def _accent_apply_labels(merged_tokens_preds):
    result = []
    for word, label_indexes in merged_tokens_preds:
        for label_idx in label_indexes:
            tag = _accent_label_list[int(label_idx)]
            raw, accented = tag.split("-")
            if raw and raw in word:
                word = word.replace(raw, accented)
                break
        result.append(word)
    return result


def restore_vietnamese_accents(text: str) -> str:
    """
    Khoi phuc dau tieng Viet tu dong cho 1 chuoi text (dung cho ocr_text).
    Ha chu thuong truoc khi dua vao model (model train tren cau van thuong, khong
    nhan tot text VIET HOA TOAN BO), roi ap lai dung case ky tu goc sau khi co ket qua.
    Model chi load 1 lan dau tien (lazy), cac lan sau dung lai. Neu load loi (mang/thieu
    dependency), tu dong bo qua buoc nay va tra ve text goc, khong lam crash pipeline.
    """
    if not ENABLE_ACCENT_RESTORE or not text or not text.strip():
        return text

    _load_accent_model()
    if _accent_load_failed:
        return text

    try:
        lowered = text.lower()
        tokens, predictions = _accent_predict(lowered)
        if not tokens:
            return text
        merged = _accent_merge(tokens, predictions)
        restored_lower = " ".join(_accent_apply_labels(merged))

        out_chars = []
        for i, rc in enumerate(restored_lower):
            out_chars.append(rc.upper() if i < len(text) and text[i].isupper() else rc)
        return "".join(out_chars)
    except Exception as e:
        print(f"WARNING: loi khi restore accent cho '{text[:50]}...': {repr(e)}")
        return text


# =========================
# GOI SERVER TU HOST (OpenAI-compatible, khong phai OpenRouter)
# =========================


def server_headers() -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if USE_OPENROUTER:
        headers["HTTP-Referer"] = "https://chatgpt.com"
        headers["X-Title"] = "AIC2026 Frame Captioning"
    return headers


def build_payload(
    image_data_url: str, prompt: str, model: str = MODEL
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    return payload


def call_server_chat_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                CHAT_COMPLETIONS_URL,
                headers=server_headers(),
                json=payload,
                timeout=TIMEOUT_SEC,
            )

            if r.status_code in {408, 409, 429, 500, 502, 503, 504}:
                last_error = f"HTTP {r.status_code}: {r.text[:500]}"
                sleep_s = min(2**attempt, 30)
                print(
                    f"Retry {attempt + 1}/{MAX_RETRIES} after {sleep_s}s because {last_error[:120]}"
                )
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_error = repr(e)
            sleep_s = min(2**attempt, 30)
            print(
                f"Retry {attempt + 1}/{MAX_RETRIES} after {sleep_s}s because {last_error[:120]}"
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Server call failed after retries: {last_error}")


def caption_one_frame(
    image_path: Path, model: str = MODEL, prompt: str = FRAME_METADATA_PROMPT
) -> Dict[str, Any]:
    original_path = image_path
    send_path = resize_image_for_api(image_path, max_side=MAX_SIDE)
    data_url = image_to_data_url(send_path)
    payload = build_payload(data_url, prompt=prompt, model=model)

    started = time.time()
    response = call_server_chat_completion(payload)
    latency_sec = round(time.time() - started, 3)

    choice0 = (response.get("choices") or [{}])[0]
    message = choice0.get("message") or {}
    raw_text = message.get("content")
    parsed = extract_json_from_text(raw_text)

    if parsed and parsed.get("ocr_text"):
        for item in parsed["ocr_text"]:
            if isinstance(item, dict) and item.get("text"):
                item["text"] = restore_vietnamese_accents(item["text"])

    usage = response.get("usage") or {}

    with Image.open(original_path) as img:
        width, height = img.size

    row = {
        "video_id": get_scene_id_for_path(original_path),
        "frame_idx": get_frame_idx_for_path(original_path),
        "timestamp_sec": get_timestamp_for_path(original_path),
        "image_path": str(original_path),
        "sent_image_path": str(send_path),
        "width": width,
        "height": height,
        "model": response.get("model", model),
        "finish_reason": choice0.get("finish_reason"),
        "usage": usage,
        "latency_sec": latency_sec,
        "raw_output": raw_text,
        "parsed": parsed,
        "parse_ok": parsed is not None,
        "error": None,
    }
    row["index_text"] = flatten_for_index(parsed) if parsed else ""
    row["hsv_features"] = extract_hsv_features(original_path)
    return row


# =========================
# SCENE-LEVEL: gui nhieu frame (cung 1 scene) trong 1 request, nhan lai metadata cho
# tung frame + 1 metadata tong hop cho ca scene.
# =========================


def build_scene_payload(
    image_data_urls: List[str], prompt: str, model: str = MODEL, max_tokens: int = MAX_TOKENS
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, data_url in enumerate(image_data_urls):
        content.append({"type": "text", "text": f"Frame {i + 1}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }


def _restore_accents_in_ocr_regions(ocr_regions: List[Dict[str, Any]]):
    """Chi khoi phuc dau cho vung OCR co language='vi' (hoac 'mixed') - khong dung cho
    text tieng Anh/khong xac dinh, tranh lam hong text dung ngon ngu khac."""
    for region in ocr_regions or []:
        if not isinstance(region, dict):
            continue
        if region.get("language") in ("vi", "mixed") and region.get("text_raw"):
            region["text_raw"] = restore_vietnamese_accents(region["text_raw"])


def flatten_keyframe_for_index(kf: Dict[str, Any]) -> str:
    """Gop cac field quan trong cua 1 keyframe (schema v2.0) thanh 1 chuoi text cho search."""
    if not kf:
        return ""
    parts = []
    for key in ("short_caption_en", "detailed_caption_en", "short_caption_vi", "detailed_caption_vi"):
        val = kf.get(key)
        if isinstance(val, str):
            parts.append(val)

    for key in ("keywords_en", "keywords_vi"):
        val = kf.get(key)
        if isinstance(val, list):
            parts.extend(str(x) for x in val if x)

    for ent in kf.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        parts.append(str(ent.get("name_en") or ""))
        parts.append(str(ent.get("description_en") or ""))
        action = ent.get("action") or {}
        if isinstance(action, dict):
            parts.append(str(action.get("label_en") or ""))
        for attr in ent.get("attributes") or []:
            if isinstance(attr, dict):
                parts.append(str(attr.get("value_en") or attr.get("name_en") or ""))

    for rel in kf.get("relations") or []:
        if isinstance(rel, dict):
            parts.append(str(rel.get("predicate_en") or ""))
            parts.append(str(rel.get("object_text_en") or ""))

    for region in kf.get("ocr_regions") or []:
        if isinstance(region, dict):
            parts.append(str(region.get("text_raw") or ""))

    return " ".join(p.strip() for p in parts if str(p).strip())


def flatten_scene_context_for_index(sc: Dict[str, Any]) -> str:
    """Gop scene_context (schema v2.0) thanh 1 chuoi text cho search."""
    if not sc:
        return ""
    parts = []
    for key in ("short_caption_en", "detailed_caption_en", "short_caption_vi", "detailed_caption_vi"):
        val = sc.get(key)
        if isinstance(val, str):
            parts.append(val)

    for ent in sc.get("scene_entities") or []:
        if isinstance(ent, dict):
            parts.append(str(ent.get("name_en") or ""))
            parts.append(str(ent.get("description_en") or ""))

    for act in sc.get("scene_actions") or []:
        if isinstance(act, dict):
            parts.append(str(act.get("label_en") or ""))

    return " ".join(p.strip() for p in parts if str(p).strip())


def caption_scene(image_paths: List[Path], model: str = MODEL) -> Dict[str, Any]:
    """
    Gui nhieu frame cung 1 scene trong DUY NHAT 1 request, dung schema day du
    "aic-multikeyframe-v2.0": tra ve "scene_context" (tong hop, song ngu, entity/action
    cross-reference giua cac keyframe) va "keyframes" (1 entry/anh, cung thu tu da gui).
    """
    num_frames = len(image_paths)
    send_paths = [resize_image_for_api(p, max_side=MAX_SIDE) for p in image_paths]
    data_urls = [image_to_data_url(p) for p in send_paths]

    prompt = build_scene_metadata_prompt(USE_COT, num_frames)
    # Schema v2.0 rat nang (song ngu + entity/action/relation/ocr long nhau). Da do thuc te:
    # voi 3 keyframe, scene_context + 1 keyframe da an het 10500 token ma chua xong (finish_
    # reason="length", JSON hong, ca scene bi bo). Tang manh budget: rieng scene_context
    # (nhieu scene_entities/scene_actions/visual_evidence) da can ~3000-4000 token, moi
    # keyframe them ~4000-6000 token tuy so entity/ocr_regions trong frame do.
    scene_max_tokens = min(32000, 4000 + num_frames * 6000)
    payload = build_scene_payload(data_urls, prompt, model=model, max_tokens=scene_max_tokens)

    started = time.time()
    response = call_server_chat_completion(payload)
    latency_sec = round(time.time() - started, 3)

    choice0 = (response.get("choices") or [{}])[0]
    message = choice0.get("message") or {}
    raw_text = message.get("content")
    parsed = extract_json_from_text(raw_text)

    keyframes_meta = (parsed or {}).get("keyframes") or []
    scene_context = (parsed or {}).get("scene_context") or {}

    for kf in keyframes_meta:
        if isinstance(kf, dict):
            _restore_accents_in_ocr_regions(kf.get("ocr_regions") or [])

    usage = response.get("usage") or {}

    per_frame_rows = []
    for i, image_path in enumerate(image_paths):
        keyframe_id = str(i + 1)
        kf = next(
            (k for k in keyframes_meta if isinstance(k, dict) and str(k.get("keyframe_id")) == keyframe_id),
            None,
        )
        if kf is None and i < len(keyframes_meta) and isinstance(keyframes_meta[i], dict):
            kf = keyframes_meta[i]
        per_frame_rows.append({
            "keyframe_id": keyframe_id,
            "video_id": get_scene_id_for_path(image_path),
            "frame_idx": get_frame_idx_for_path(image_path),
            "timestamp_sec": get_timestamp_for_path(image_path),
            "image_path": str(image_path),
            "parsed": kf,
            "parse_ok": kf is not None,
            "index_text": flatten_keyframe_for_index(kf) if kf else "",
            "hsv_features": extract_hsv_features(image_path),
        })

    return {
        "schema_version": SCENE_SCHEMA_VERSION,
        "scene_key": get_scene_id_for_path(image_paths[0]) if image_paths else None,
        "image_paths": [str(p) for p in image_paths],
        "num_frames": num_frames,
        "model": response.get("model", model),
        "finish_reason": choice0.get("finish_reason"),
        "usage": usage,
        "latency_sec": latency_sec,
        "raw_output": raw_text,
        "parse_ok": parsed is not None,
        "keyframes": per_frame_rows,
        "scene_context": scene_context,
        "scene_index_text": flatten_scene_context_for_index(scene_context) if scene_context else "",
        "error": None,
    }


# =========================
# GHI/DOC JSON ARRAY (resume-able, giong notebook)
# =========================


def read_json_array(path: Path) -> List[Dict[str, Any]]:
    """Doc file JSONL (1 JSON object/dong). Ten ham giu nguyen de khong doi cac cho
    goi khac trong code, nhung tu gio doc/ghi theo dinh dang JSONL, khong phai JSON array."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json_array(path: Path, rows: List[Dict[str, Any]]):
    """Ghi de toan bo thanh JSONL (dung cho export_*_index_ready, chi ghi 1 lan cuoi)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_image_paths(json_path: Path) -> set:
    rows = read_json_array(json_path)
    return {row["image_path"] for row in rows if row.get("image_path")}


def append_json_row(path: Path, row: Dict[str, Any]):
    """Append 1 dong JSONL - O(1), khong can doc lai toan bo file nhu truoc (JSON array
    phai doc-sua-ghi lai ca mang moi lan append, cham dan khi file lon o quy mo 10-80k)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# BATCH CAPTION SONG SONG (khac notebook: co ThreadPoolExecutor + Lock)
# =========================


def caption_frames_batch_parallel(
    frames_dir: Path,
    out_json: Path,
    failed_json: Path,
    limit: Optional[int] = None,
    max_workers: int = 8,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    images = list_images(frames_dir)
    if limit is not None:
        images = images[:limit]

    done = load_done_image_paths(out_json)
    pending = [p for p in images if str(p) not in done]

    print("frames_dir:", frames_dir)
    print("total images selected:", len(images))
    print("already done:", len(done))
    print("pending:", len(pending))
    print("out_json:", out_json)
    print("max_workers:", max_workers)

    write_lock = threading.Lock()

    def _process(image_path: Path):
        try:
            row = caption_one_frame(image_path, model=MODEL)
            with write_lock:
                append_json_row(out_json, row)
        except Exception as e:
            err_row = {
                "video_id": get_scene_id_for_path(image_path),
                "frame_idx": get_frame_idx_for_path(image_path),
                "timestamp_sec": get_timestamp_for_path(image_path),
                "image_path": str(image_path),
                "model": MODEL,
                "parse_ok": False,
                "parsed": None,
                "raw_output": None,
                "error": repr(e),
                "traceback": traceback.format_exc()[-4000:],
            }
            with write_lock:
                append_json_row(failed_json, err_row)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, p) for p in pending]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Caption frames"):
            pass


def export_index_ready(in_json: Path, out_json: Path):
    rows = read_json_array(in_json)
    out_rows = []
    for r in rows:
        parsed = r.get("parsed") or {}
        if not parsed:
            continue
        out_rows.append(
            {
                "video_id": r.get("video_id"),
                "frame_idx": r.get("frame_idx"),
                "timestamp_sec": r.get("timestamp_sec"),
                "image_path": r.get("image_path"),
                "model": r.get("model"),
                "short_caption": parsed.get("short_caption"),
                "detailed_caption": parsed.get("detailed_caption"),
                "scene_type": parsed.get("scene_type"),
                "objects": parsed.get("objects"),
                "actions": parsed.get("actions"),
                "attributes": parsed.get("attributes"),
                "relations": parsed.get("relations"),
                "ocr_text": parsed.get("ocr_text"),
                "search_keywords": parsed.get("search_keywords"),
                "main_subjects": parsed.get("main_subjects"),
                "people": parsed.get("people"),
                "retrieval_notes": parsed.get("retrieval_notes"),
                "index_text": r.get("index_text") or flatten_for_index(parsed),
                "hsv_features": r.get("hsv_features"),
            }
        )
    write_json_array(out_json, out_rows)
    print(f"Exported {len(out_rows)} rows to {out_json}")


# =========================
# BATCH CAPTION SCENE-LEVEL SONG SONG (moi scene = 1 request nhieu anh)
# =========================


def caption_scenes_batch_parallel(
    frames_dir: Path,
    out_json: Path,
    failed_json: Path,
    limit: Optional[int] = None,
    max_workers: int = 8,
    max_frames_per_scene: int = MAX_FRAMES_PER_SCENE,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    images = list_images(frames_dir)
    scene_groups = group_frames_by_scene(images)
    scene_keys = sorted(scene_groups.keys())
    if limit is not None:
        scene_keys = scene_keys[:limit]

    done_keys = {row.get("scene_key") for row in read_json_array(out_json)}
    pending_keys = [k for k in scene_keys if k not in done_keys]

    print("frames_dir:", frames_dir)
    print("total scenes:", len(scene_keys))
    print("already done:", len(done_keys))
    print("pending:", len(pending_keys))
    print("out_json:", out_json)
    print("max_workers:", max_workers)
    print("max_frames_per_scene:", max_frames_per_scene)

    write_lock = threading.Lock()

    def _process(scene_key: str):
        paths = sample_frames_even(scene_groups[scene_key], max_frames_per_scene)
        try:
            row = caption_scene(paths, model=MODEL)
            row["scene_key"] = scene_key
            with write_lock:
                append_json_row(out_json, row)
        except Exception as e:
            err_row = {
                "scene_key": scene_key,
                "image_paths": [str(p) for p in paths],
                "parse_ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc()[-4000:],
            }
            with write_lock:
                append_json_row(failed_json, err_row)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, k) for k in pending_keys]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Caption scenes"):
            pass


def export_scene_index_ready(in_json: Path, out_json: Path):
    """
    Lam phang ket qua scene-level (schema aic-multikeyframe-v2.0) thanh cac dong rieng:
    1 dong/keyframe + 1 dong tong hop cho ca scene (record_type = "keyframe" hoac
    "scene_context"), du de dua vao BM25/embedding. Giu nguyen cau truc parsed day du
    (entities/relations/ocr_regions...) trong field "raw_parsed" cho ai can chi tiet sau.
    """
    rows = read_json_array(in_json)
    out_rows = []
    for r in rows:
        scene_key = r.get("scene_key")
        model = r.get("model")

        for kf_row in r.get("keyframes") or []:
            parsed = kf_row.get("parsed") or {}
            if not parsed:
                continue

            ocr_regions = parsed.get("ocr_regions") or []
            entities = parsed.get("entities") or []
            retry_flags = [
                reg.get("region_id")
                for reg in ocr_regions
                if isinstance(reg, dict) and reg.get("model_retry_recommended")
            ]

            out_rows.append({
                "record_type": "keyframe",
                "scene_key": scene_key,
                "keyframe_id": kf_row.get("keyframe_id"),
                "video_id": kf_row.get("video_id"),
                "frame_idx": kf_row.get("frame_idx"),
                "timestamp_sec": kf_row.get("timestamp_sec"),
                "image_path": kf_row.get("image_path"),
                "model": model,
                "short_caption_en": parsed.get("short_caption_en"),
                "short_caption_vi": parsed.get("short_caption_vi"),
                "detailed_caption_en": parsed.get("detailed_caption_en"),
                "detailed_caption_vi": parsed.get("detailed_caption_vi"),
                "entities": entities,
                "relations": parsed.get("relations"),
                "ocr_regions": ocr_regions,
                "ocr_retry_recommended_region_ids": retry_flags,
                "keywords_en": parsed.get("keywords_en"),
                "keywords_vi": parsed.get("keywords_vi"),
                "frame_selection": parsed.get("frame_selection"),
                "uncertainties": parsed.get("uncertainties"),
                "index_text": kf_row.get("index_text") or flatten_keyframe_for_index(parsed),
                "hsv_features": kf_row.get("hsv_features"),
            })

        scene_context = r.get("scene_context") or {}
        if scene_context:
            out_rows.append({
                "record_type": "scene_context",
                "scene_key": scene_key,
                "keyframe_id": None,
                "video_id": scene_key,
                "frame_idx": None,
                "timestamp_sec": None,
                "image_path": None,
                "model": model,
                "environment": scene_context.get("environment"),
                "setting": scene_context.get("setting"),
                "media_type": scene_context.get("media_type"),
                "short_caption_en": scene_context.get("short_caption_en"),
                "short_caption_vi": scene_context.get("short_caption_vi"),
                "detailed_caption_en": scene_context.get("detailed_caption_en"),
                "detailed_caption_vi": scene_context.get("detailed_caption_vi"),
                "scene_entities": scene_context.get("scene_entities"),
                "visual_evidence": scene_context.get("visual_evidence"),
                "scene_actions": scene_context.get("scene_actions"),
                "best_keyframe_id": scene_context.get("best_keyframe_id"),
                "best_keyframe_reason_en": scene_context.get("best_keyframe_reason_en"),
                "best_keyframe_reason_vi": scene_context.get("best_keyframe_reason_vi"),
                "uncertainties": scene_context.get("uncertainties"),
                "index_text": r.get("scene_index_text") or flatten_scene_context_for_index(scene_context),
            })

    write_json_array(out_json, out_rows)
    print(f"Exported {len(out_rows)} rows to {out_json}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_RESIZED_DIR.mkdir(parents=True, exist_ok=True)

    print("MODEL:", MODEL)
    print("SERVER_BASE_URL:", SERVER_BASE_URL)
    print("FRAMES_DIR:", FRAMES_DIR)
    print("RUN_MODE:", RUN_MODE)
    print("LIMIT:", LIMIT)
    print("MAX_WORKERS:", MAX_WORKERS)
    print("-" * 60)

    if RUN_MODE == "scene":
        print("OUT_SCENE_JSON:", OUT_SCENE_JSON)
        print("MAX_FRAMES_PER_SCENE:", MAX_FRAMES_PER_SCENE)
        print("-" * 60)

        caption_scenes_batch_parallel(
            FRAMES_DIR,
            OUT_SCENE_JSON,
            OUT_SCENE_FAILED_JSON,
            limit=LIMIT,
            max_workers=MAX_WORKERS,
            max_frames_per_scene=MAX_FRAMES_PER_SCENE,
        )
        export_scene_index_ready(OUT_SCENE_JSON, OUT_SCENE_INDEX_JSON)

        rows = read_json_array(OUT_SCENE_JSON)
        ok = sum(1 for r in rows if r.get("parse_ok"))
        print("-" * 60)
        print(
            f"Tong so scene da xu ly: {len(rows)}, parse_ok: {ok} ({ok/len(rows)*100:.1f}%)"
            if rows
            else "Chua co scene nao."
        )
    else:
        print("OUT_JSON:", OUT_JSON)
        print("-" * 60)

        caption_frames_batch_parallel(
            FRAMES_DIR,
            OUT_JSON,
            OUT_FAILED_JSON,
            limit=LIMIT,
            max_workers=MAX_WORKERS,
        )
        export_index_ready(OUT_JSON, OUT_INDEX_JSON)

        rows = read_json_array(OUT_JSON)
        ok = sum(1 for r in rows if r.get("parse_ok"))
        print("-" * 60)
        print(
            f"Tong so anh da xu ly: {len(rows)}, parse_ok: {ok} ({ok/len(rows)*100:.1f}%)"
            if rows
            else "Chua co anh nao."
        )
