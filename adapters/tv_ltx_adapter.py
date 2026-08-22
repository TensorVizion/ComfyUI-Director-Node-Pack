"""
TV Director -> LTX adapter.

Consumes a DIRECTOR_TIMELINE (from TVDirectorTimeline) and produces the
conditioning objects LTXVideo sampling expects, using ComfyUI-LTXVideo's
native nodes under the hood (LTXVConditioning-style positive/negative
conditioning plus per-shot guide latents via LTXVAddGuide).

Requires ComfyUI-LTXVideo to be installed, since we import and call its
node classes directly rather than reimplementing LTX's guide-conditioning
math. This mirrors how LTX Director itself builds on top of the same
node pack instead of hand-rolling LTX internals.
"""

from __future__ import annotations

from ..timeline_schema import validate_timeline, ordered_shots, resolve_prompt


def _get_ltxvideo_nodes():
    """
    Import ComfyUI-LTXVideo's node classes at call time (not import time)
    so this file can be loaded even if LTXVideo isn't installed yet — the
    adapter node will just raise a clear error when actually executed.
    """
    try:
        from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODES
    except ImportError as e:
        raise RuntimeError(
            "Could not access ComfyUI's global node registry."
        ) from e

    required = ["LTXVConditioning", "LTXVAddGuide", "LTXVBaseSampler"]
    missing = [n for n in required if n not in GLOBAL_NODES]
    if missing:
        raise RuntimeError(
            "TV->LTX Adapter requires ComfyUI-LTXVideo to be installed and "
            f"up to date. Missing node classes: {missing}. Install/update "
            "via ComfyUI Manager: https://github.com/Lightricks/ComfyUI-LTXVideo"
        )
    return {name: GLOBAL_NODES[name] for name in required}


class TVToLTXAdapter:
    """
    Translates a generic DIRECTOR_TIMELINE into LTX-native positive/negative
    conditioning plus a guide-latent stack, ready to feed into LTXVBaseSampler
    or a KSampler configured for an LTX checkpoint.
    """

    NODE_NAME = "TV → LTX Adapter 🎬"
    CATEGORY = "TensorVizion/Director/Adapters"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("DIRECTOR_TIMELINE",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "guide_latent", "num_frames")
    FUNCTION = "adapt"
    CATEGORY = CATEGORY

    def adapt(self, timeline, clip, vae):
        errors = validate_timeline(timeline)
        if errors:
            raise ValueError("Cannot adapt invalid timeline for LTX:\n  - " + "\n  - ".join(errors))

        ltx_nodes = _get_ltxvideo_nodes()
        conditioning_node = ltx_nodes["LTXVConditioning"]()
        add_guide_node = ltx_nodes["LTXVAddGuide"]()

        shots = ordered_shots(timeline)
        images = timeline.get("_runtime", {}).get("images")
        fps = timeline.get("global", {}).get("fps", 24)

        positive_acc = None
        negative_acc = None
        guide_latent = None
        total_frames = 0

        for shot in shots:
            prompt, negative = resolve_prompt(timeline, shot)

            # Text-encode this shot's prompt via the standard CLIP encode
            # call, then hand it to LTXVConditioning for frame-rate-aware
            # conditioning, matching how LTXVideo workflows are normally wired.
            pos_cond = _clip_encode(clip, prompt)
            neg_cond = _clip_encode(clip, negative or "")

            pos_cond, neg_cond = conditioning_node.append(
                positive=pos_cond,
                negative=neg_cond,
                frame_rate=fps,
            )

            positive_acc = _merge_conditioning(positive_acc, pos_cond)
            negative_acc = _merge_conditioning(negative_acc, neg_cond)

            # Resolve this shot's reference image (if any) and inject it
            # as a guide frame at the correct position/strength.
            img_tensor = _resolve_image_ref(images, shot.get("image_ref"))
            if img_tensor is not None and guide_latent is not None:
                frame_idx = _guide_frame_index(shot, total_frames)
                (guide_latent,) = add_guide_node.generate(
                    latent=guide_latent,
                    vae=vae,
                    image=img_tensor,
                    frame_idx=frame_idx,
                    strength=shot.get("strength", 1.0),
                )
            elif img_tensor is not None and guide_latent is None:
                # First guide image seeds the latent stack.
                guide_latent = _encode_first_guide(vae, img_tensor)

            total_frames += int(shot.get("duration_frames", 0))

        if guide_latent is None:
            # Pure T2V timeline with no reference images: build an empty
            # latent sized to the requested resolution / frame count.
            guide_latent = _empty_video_latent(timeline, total_frames)

        return (positive_acc, negative_acc, guide_latent, total_frames)


# --- helpers ------------------------------------------------------------

def _clip_encode(clip, text):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def _merge_conditioning(acc, new):
    if acc is None:
        return new
    return acc + new


def _resolve_image_ref(images, ref):
    if images is None or ref is None:
        return None
    try:
        return images[ref:ref + 1]
    except (IndexError, TypeError):
        return None


def _guide_frame_index(shot, running_total):
    role = shot.get("image_role", "first")
    if role == "first":
        return running_total
    if role == "last":
        return running_total + int(shot.get("duration_frames", 0)) - 1
    # "middle" / "reference": place at shot midpoint as a reasonable default
    return running_total + int(shot.get("duration_frames", 0)) // 2


def _encode_first_guide(vae, img_tensor):
    encoded = vae.encode(img_tensor[:, :, :, :3])
    return {"samples": encoded}


def _empty_video_latent(timeline, total_frames):
    import torch
    g = timeline.get("global", {})
    w = g.get("width", 768) // 8
    h = g.get("height", 512) // 8
    t = max(1, total_frames // 8 + 1)
    return {"samples": torch.zeros([1, 128, t, h, w])}


NODE_CLASS_MAPPINGS = {
    "TVToLTXAdapter": TVToLTXAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TVToLTXAdapter": "TV → LTX Adapter 🎬",
}
