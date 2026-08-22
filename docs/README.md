# TensorVizion Director 🎬

A model-agnostic timeline node for ComfyUI, inspired by LTX Director's
drag-and-drop shot editor — but the timeline UI itself doesn't know or
care which video model you're using. Backend-specific **adapter nodes**
translate the timeline into native conditioning calls for **LTX**,
**Wan**, or **Hunyuan**.

## Why adapters instead of one universal node

LTX, Wan, and Hunyuan condition video generation in genuinely different
ways:

| Backend | Conditioning model |
|---|---|
| LTX | One continuous guide-latent stack; images injected at arbitrary frame indices via `LTXVAddGuide` |
| Wan | Per-clip: `WanImageToVideo` (single start image) or `WanFirstLastFrameToVideo` (start+end pair), one conditioning+latent triple per clip |
| Hunyuan | Per-clip: `TextEncodeHunyuanVideo_ImageToVideo` + `EmptyHunyuanLatentVideo`, similar shape to Wan |

A single node pretending to handle all three identically would either be
shallow (lowest common denominator: text + one image, nothing else) or
silently wrong for two of the three backends. Instead:

- **TV Director** — the timeline UI. Build your shot list once: prompts,
  reference images, durations, transitions, camera hints, audio refs.
  Outputs a `DIRECTOR_TIMELINE` object (JSON under the hood) — a
  documented, versioned schema (see `timeline_schema.py`).
- **TV → LTX Adapter** — flattens the timeline into one positive/negative
  conditioning pair + one guide latent, matching how LTX actually samples.
- **TV → Wan Adapter** — returns a list of per-shot conditioning bundles
  (auto-pairs adjacent "first"/"last" shots into `WanFirstLastFrameToVideo`
  calls when both have images).
- **TV → Hunyuan Adapter** — same per-shot list shape as Wan, using
  Hunyuan's native I2V/T2V encode nodes.
- **TV Wan Shot Iterator** — pulls one shot's bundle out of a Wan or
  Hunyuan shot list by index, for feeding into a sampler in a per-shot
  loop.

## Install

```
cd ComfyUI/custom_nodes/
git clone <your-repo-url> tensorvizion-director
```

Then install/update the backend node packs for whichever models you plan
to use:
- **LTX**: [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) (required for the LTX adapter — it calls `LTXVConditioning` / `LTXVAddGuide` directly rather than reimplementing them)
- **Wan**: ComfyUI core, updated to a version with Wan 2.x support (`WanImageToVideo` / `WanFirstLastFrameToVideo` ship as built-in nodes)
- **Hunyuan**: ComfyUI core with the classic HunyuanVideo node set (`TextEncodeHunyuanVideo_ImageToVideo`, `EmptyHunyuanLatentVideo`)

Each adapter checks for its required nodes at execution time and raises a
clear error naming exactly what's missing — it won't fail silently or
produce garbage output if a backend isn't installed.

## Known limitations (read before reporting a bug)

- **Hunyuan 1.5** ships different node names than classic HunyuanVideo
  (e.g. `HunyuanVideo15ImageToVideo`). This adapter targets classic
  HunyuanVideo only; 1.5 support would need its own adapter variant.
- **Transitions** (`crossfade`, `hold`, `morph`) are stored in the
  timeline schema as *intent* but none of the three adapters currently
  render them — all shots are conditioned independently and concatenated.
  Actual crossfade/morph blending between shots is a real feature gap,
  not an oversight to paper over.
- **Wan/Hunyuan multi-shot stitching** is manual: you sample each shot
  separately and concatenate the resulting video clips yourself with a
  video-combine node. There's no automatic continuity-preserving stitch
  (e.g. carrying latent noise state between shots) yet.
- Built and tested against node APIs current as of mid-2026; if
  `LTXVAddGuide`, `WanImageToVideo`, or the Hunyuan encode nodes change
  their signatures upstream, the adapters will need matching updates.

## Timeline schema

See `timeline_schema.py` for the full, documented schema (`SCHEMA_VERSION
= 1`). Key shape:

```json
{
  "schema_version": 1,
  "global": { "fps": 24, "width": 768, "height": 512, "seed": -1, ... },
  "audio_tracks": [...],
  "shots": [
    {
      "id": "shot_abc123",
      "order": 0,
      "prompt": "...",
      "negative_prompt": "...",
      "duration_frames": 97,
      "image_ref": 0,
      "image_role": "first",
      "transition_in": "cut",
      "transition_out": "cut",
      "strength": 1.0,
      "camera_hint": "slow dolly in"
    }
  ]
}
```
