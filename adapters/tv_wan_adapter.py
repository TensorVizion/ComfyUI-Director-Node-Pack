"""
TV Director -> Wan adapter.

Wan's conditioning model differs fundamentally from LTX's guide-latent
stack: Wan builds video conditioning per-clip via WanImageToVideo (single
start image) or WanFirstLastFrameToVideo (start+end image pair), each
producing its own positive/negative/latent triple rather than one
continuously-injected guide latent.

Because of that, this adapter treats each DIRECTOR_TIMELINE shot as its
own Wan clip: it looks at the shot's image_role to decide whether to call
WanImageToVideo (role "first" alone) or WanFirstLastFrameToVideo (a
"first"+"last" pair), then returns a LIST of per-shot conditioning
triples for a downstream sampler-per-shot loop, rather than pretending
Wan can be flattened into one global conditioning object the way LTX can.

Requires ComfyUI's built-in Wan video nodes (ships with core ComfyUI as
of the Wan 2.x integration).
"""

from __future__ import annotations

from ..timeline_schema import validate_timeline, ordered_shots, resolve_prompt


def _get_wan_nodes():
    try:
        from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODES
    except ImportError as e:
        raise RuntimeError("Could not access ComfyUI's global node registry.") from e

    required = ["WanImageToVideo", "WanFirstLastFrameToVideo"]
    missing = [n for n in required if n not in GLOBAL_NODES]
    if missing:
        raise RuntimeError(
            "TV->Wan Adapter requires ComfyUI's built-in Wan video nodes. "
            f"Missing: {missing}. Update ComfyUI to a version with Wan 2.x "
            "support (WanImageToVideo / WanFirstLastFrameToVideo)."
        )
    return {name: GLOBAL_NODES[name] for name in required}


class TVToWanAdapter:
    """
    Translates a DIRECTOR_TIMELINE into a list of per-shot Wan conditioning
    bundles. Each item is {positive, negative, latent, num_frames, shot_id}.
    Feed this list into a companion "TV Wan Shot Iterator" or loop it with
    a standard ComfyUI list-processing pattern into KSampler per shot,
    then concatenate the resulting video latents downstream.
    """

    NODE_NAME = "TV → Wan Adapter 🎬"
    CATEGORY = "TensorVizion/Director/Adapters"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("DIRECTOR_TIMELINE",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
            },
        }

    RETURN_TYPES = ("TV_WAN_SHOTLIST", "INT")
    RETURN_NAMES = ("wan_shots", "shot_count")
    FUNCTION = "adapt"
    CATEGORY = CATEGORY

    def adapt(self, timeline, clip, vae, clip_vision=None):
        errors = validate_timeline(timeline)
        if errors:
            raise ValueError("Cannot adapt invalid timeline for Wan:\n  - " + "\n  - ".join(errors))

        wan_nodes = _get_wan_nodes()
        i2v_node = wan_nodes["WanImageToVideo"]()
        flf2v_node = wan_nodes["WanFirstLastFrameToVideo"]()

        shots = ordered_shots(timeline)
        images = timeline.get("_runtime", {}).get("images")
        g = timeline.get("global", {})
        width, height = g.get("width", 832), g.get("height", 480)

        results = []
        i = 0
        while i < len(shots):
            shot = shots[i]
            prompt, negative = resolve_prompt(timeline, shot)
            pos_cond = _clip_encode(clip, prompt)
            neg_cond = _clip_encode(clip, negative or "")
            length = int(shot.get("duration_frames", 81))

            start_img = _resolve_image_ref(images, shot.get("image_ref"))
            role = shot.get("image_role", "first")

            # Look ahead: if this shot is a "first" and the very next shot
            # is a "last" with its own image, treat them as one FLF2V clip
            # rather than two separate I2V clips. This matches how a
            # director would naturally lay out a start/end pair on the
            # timeline as two adjacent cards.
            paired_last = None
            if role == "first" and i + 1 < len(shots) and shots[i + 1].get("image_role") == "last":
                paired_last = shots[i + 1]

            if paired_last is not None:
                end_img = _resolve_image_ref(images, paired_last.get("image_ref"))
                if end_img is not None and start_img is not None:
                    pos, neg, latent = flf2v_node.encode(
                        positive=pos_cond,
                        negative=neg_cond,
                        vae=vae,
                        width=width,
                        height=height,
                        length=length,
                        batch_size=1,
                        start_image=start_img,
                        end_image=end_img,
                        clip_vision_start_image=clip_vision,
                        clip_vision_end_image=clip_vision,
                    )
                    results.append({
                        "shot_id": shot["id"],
                        "positive": pos,
                        "negative": neg,
                        "latent": latent,
                        "num_frames": length,
                    })
                    i += 2
                    continue

            # Default path: single-image (or pure T2V) I2V clip.
            pos, neg, latent = i2v_node.encode(
                positive=pos_cond,
                negative=neg_cond,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                start_image=start_img,
                clip_vision_output=clip_vision,
            )
            results.append({
                "shot_id": shot["id"],
                "positive": pos,
                "negative": neg,
                "latent": latent,
                "num_frames": length,
            })
            i += 1

        return (results, len(results))


def _clip_encode(clip, text):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def _resolve_image_ref(images, ref):
    if images is None or ref is None:
        return None
    try:
        return images[ref:ref + 1]
    except (IndexError, TypeError):
        return None


NODE_CLASS_MAPPINGS = {
    "TVToWanAdapter": TVToWanAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TVToWanAdapter": "TV → Wan Adapter 🎬",
}
