"""
TensorVizion Director — shared timeline schema.

This is the contract between the TV Director timeline node (frontend UI,
model-agnostic) and every backend adapter (TV->LTX, TV->Wan, TV->Hunyuan).

The timeline node never talks to LTX/Wan/Hunyuan directly. It only ever
produces a DIRECTOR_TIMELINE dict matching SCHEMA_VERSION below. Adapters
are the only place that translates this into native conditioning calls.

Keeping this in one file means all three adapters and the UI node agree
on field names without copy-pasted drift.
"""

from __future__ import annotations
import json
import copy

SCHEMA_VERSION = 1

# Transition types are intentionally backend-agnostic descriptions of
# *intent*. Not every backend can honor every transition; adapters fall
# back to "cut" with a console warning if they can't support one.
VALID_TRANSITIONS = {"cut", "crossfade", "hold", "morph"}


def empty_timeline(fps: int = 24, width: int = 768, height: int = 512) -> dict:
    """Return a fresh, valid, empty timeline dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "global": {
            "fps": fps,
            "width": width,
            "height": height,
            "seed": -1,
            "global_prompt_prefix": "",
            "global_negative_prompt": "",
        },
        "audio_tracks": [],  # list of {id, path, start_frame, trim_start, trim_end}
        "shots": [],
    }


def new_shot(shot_id: str) -> dict:
    """Return a single blank shot dict with every field the adapters expect."""
    return {
        "id": shot_id,
        "order": 0,
        "prompt": "",
        "negative_prompt": "",
        "duration_frames": 97,          # ~4s @ 24fps default, matches LTX conventions
        "image_ref": None,               # base64 / IMAGE tensor ref index, or None for T2V
        "image_role": "first",           # "first" | "last" | "middle" | "reference"
        "transition_in": "cut",
        "transition_out": "cut",
        "audio_track_id": None,
        "strength": 1.0,                 # conditioning strength, 0-1
        "camera_hint": "",                # free text: "slow dolly in", "static", etc.
    }


def validate_timeline(timeline: dict) -> list[str]:
    """
    Validate a DIRECTOR_TIMELINE dict. Returns a list of human-readable
    problems; empty list means valid. Adapters should call this first and
    raise/refuse cleanly rather than guessing at malformed input.
    """
    errors = []

    if not isinstance(timeline, dict):
        return ["Timeline is not a dict/object."]

    if timeline.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version {timeline.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}. Re-save the timeline in TV Director."
        )

    g = timeline.get("global")
    if not isinstance(g, dict):
        errors.append("Missing 'global' section.")
    else:
        for key in ("fps", "width", "height"):
            if key not in g:
                errors.append(f"global.{key} is missing.")

    shots = timeline.get("shots")
    if not isinstance(shots, list) or len(shots) == 0:
        errors.append("Timeline has no shots. Add at least one shot.")
    else:
        seen_ids = set()
        for i, shot in enumerate(shots):
            prefix = f"shot[{i}]"
            if not isinstance(shot, dict):
                errors.append(f"{prefix} is not an object.")
                continue
            sid = shot.get("id")
            if not sid:
                errors.append(f"{prefix} missing 'id'.")
            elif sid in seen_ids:
                errors.append(f"{prefix} duplicate id {sid!r}.")
            else:
                seen_ids.add(sid)

            dur = shot.get("duration_frames")
            if not isinstance(dur, (int, float)) or dur <= 0:
                errors.append(f"{prefix} duration_frames must be > 0, got {dur!r}.")

            role = shot.get("image_role", "first")
            if role not in ("first", "last", "middle", "reference"):
                errors.append(f"{prefix} invalid image_role {role!r}.")

            t_in = shot.get("transition_in", "cut")
            t_out = shot.get("transition_out", "cut")
            if t_in not in VALID_TRANSITIONS:
                errors.append(f"{prefix} invalid transition_in {t_in!r}.")
            if t_out not in VALID_TRANSITIONS:
                errors.append(f"{prefix} invalid transition_out {t_out!r}.")

    return errors


def ordered_shots(timeline: dict) -> list[dict]:
    """Return shots sorted by their 'order' field (stable)."""
    shots = copy.deepcopy(timeline.get("shots", []))
    shots.sort(key=lambda s: s.get("order", 0))
    return shots


def total_frames(timeline: dict) -> int:
    return sum(int(s.get("duration_frames", 0)) for s in timeline.get("shots", []))


def resolve_prompt(timeline: dict, shot: dict) -> tuple[str, str]:
    """
    Merge the global prompt prefix/negative with a shot's own prompt.
    Every adapter should call this instead of reading shot['prompt']
    directly, so global prefixing behaves identically across backends.
    """
    g = timeline.get("global", {})
    prefix = g.get("global_prompt_prefix", "").strip()
    shot_prompt = shot.get("prompt", "").strip()
    full_prompt = f"{prefix}, {shot_prompt}" if prefix and shot_prompt else (prefix or shot_prompt)

    global_neg = g.get("global_negative_prompt", "").strip()
    shot_neg = shot.get("negative_prompt", "").strip()
    full_neg = f"{global_neg}, {shot_neg}" if global_neg and shot_neg else (global_neg or shot_neg)

    return full_prompt, full_neg


def loads(json_str: str) -> dict:
    return json.loads(json_str)


def dumps(timeline: dict) -> str:
    return json.dumps(timeline, indent=2)
