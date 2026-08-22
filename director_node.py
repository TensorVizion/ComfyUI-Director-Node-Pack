"""
TV Director — model-agnostic timeline node.

This node has no knowledge of LTX / Wan / Hunyuan. It owns exactly one job:
let the user build a shot-by-shot timeline (prompts, images, durations,
transitions, audio) and emit it as a DIRECTOR_TIMELINE object. Backend
adapters (tv_ltx_adapter.py, tv_wan_adapter.py, tv_hunyuan_adapter.py)
consume that object and do the actual model-specific conditioning.

The heavy lifting UI (drag/drop reordering, per-shot cards, timeline
scrubber) lives in js/tv_director.js. This Python side just:
  - declares the node's inputs/outputs for the ComfyUI graph
  - stores the serialized timeline JSON in a hidden widget
  - validates on execution so broken timelines fail with a clear message
    instead of silently producing garbage for the adapters
"""

from __future__ import annotations
import torch

from .timeline_schema import (
    validate_timeline,
    loads as timeline_loads,
    empty_timeline,
)


class TVDirectorTimeline:
    """
    Main timeline editor node. The `timeline_json` widget is populated and
    edited entirely by the custom JS UI; it is hidden from the normal
    ComfyUI widget stack via the frontend's registered widget type
    ("TVDIRECTOR_TIMELINE") so it doesn't render as a raw text box.
    """

    NODE_NAME = "TV Director 🎬"
    CATEGORY = "TensorVizion/Director"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_json": ("STRING", {
                    "default": timeline_loads_default(),
                    "multiline": True,
                    "tvdirector_widget": True,  # signals the JS to replace this with the timeline UI
                }),
            },
            "optional": {
                # Optional IMAGE batch input lets users wire an upstream
                # Load Image / Multi Image Loader node in and reference
                # frames from the timeline UI by index.
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("DIRECTOR_TIMELINE", "INT", "STRING")
    RETURN_NAMES = ("timeline", "total_frames", "summary")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, timeline_json, images=None, audio=None):
        timeline = timeline_loads(timeline_json)

        errors = validate_timeline(timeline)
        if errors:
            raise ValueError(
                "TV Director timeline is invalid:\n  - " + "\n  - ".join(errors)
            )

        # Attach the raw upstream tensors so adapters can resolve
        # image_ref / audio_track_id indices into actual tensors without
        # the timeline node needing to know what those tensors are for.
        timeline["_runtime"] = {
            "images": images,
            "audio": audio,
        }

        total = sum(int(s.get("duration_frames", 0)) for s in timeline.get("shots", []))
        n_shots = len(timeline.get("shots", []))
        fps = timeline.get("global", {}).get("fps", 24)
        seconds = total / fps if fps else 0
        summary = f"{n_shots} shots, {total} frames (~{seconds:.1f}s @ {fps}fps)"

        return (timeline, total, summary)


def timeline_loads_default() -> str:
    import json
    return json.dumps(empty_timeline())


NODE_CLASS_MAPPINGS = {
    "TVDirectorTimeline": TVDirectorTimeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TVDirectorTimeline": "TV Director 🎬",
}
