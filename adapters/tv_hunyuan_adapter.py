"""
TV Director -> Hunyuan adapter.

HunyuanVideo's conditioning shape is closer to Wan's per-clip model than
LTX's continuous guide stack: TextEncodeHunyuanVideo_ImageToVideo bundles
text + a single reference image into one conditioning object, and
EmptyHunyuanLatentVideo sizes the initial latent for a given frame count.
There's no native multi-shot guide injection like LTXVAddGuide, so — same
as the Wan adapter — this returns one conditioning+latent bundle per shot
rather than a single flattened conditioning object.

Note: HunyuanVideo 1.5 ships its own node names (e.g.
HunyuanVideo15ImageToVideo) that aren't guaranteed compatible with the
classic HunyuanVideo I2V node encoded here. This adapter targets the
classic/most widely distributed HunyuanVideo node set
(TextEncodeHunyuanVideo_ImageToVideo + EmptyHunyuanLatentVideo). If you're
on 1.5, check your installed node names first — the adapter will raise a
clear error if the classic nodes aren't present rather than silently
picking the wrong ones.
"""

from __future__ import annotations

from ..timeline_schema import validate_timeline, ordered_shots, resolve_prompt


def _get_hunyuan_nodes():
    try:
        from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODES
    except ImportError as e:
        raise RuntimeError("Could not access ComfyUI's global node registry.") from e

    required = ["TextEncodeHunyuanVideo_ImageToVideo", "EmptyHunyuanLatentVideo", "CLIPTextEncode"]
    missing = [n for n in required if n not in GLOBAL_NODES]
    if missing:
        raise RuntimeError(
            "TV->Hunyuan Adapter requires the classic HunyuanVideo node set "
            f"in ComfyUI core. Missing: {missing}. If you're running "
            "HunyuanVideo 1.5, its node names differ (e.g. "
            "HunyuanVideo15ImageToVideo) and this adapter does not yet "
            "target that variant — update ComfyUI or check your installed "
            "node names before retrying."
        )
    return {name: GLOBAL_NODES[name] for name in required}


class TVToHunyuanAdapter:
    """
    Translates a DIRECTOR_TIMELINE into a list of per-shot HunyuanVideo
    conditioning bundles: {positive, negative, latent, num_frames, shot_id}.
    Pair with TVWanShotIterator (schema-compatible) to pull one shot at a
    time into a sampler, since Hunyuan's per-clip conditioning model needs
    the same shot-by-shot loop as Wan rather than one flattened timeline.
    """

    NODE_NAME = "TV → Hunyuan Adapter 🎬"
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

    RETURN_TYPES = ("TV_WAN_SHOTLIST", "INT")  # same shape as Wan's list; iterator is reusable
    RETURN_NAMES = ("hunyuan_shots", "shot_count")
    FUNCTION = "adapt"
    CATEGORY = CATEGORY

    def adapt(self, timeline, clip, vae, clip_vision=None):
        errors = validate_timeline(timeline)
        if errors:
            raise ValueError("Cannot adapt invalid timeline for Hunyuan:\n  - " + "\n  - ".join(errors))

        nodes = _get_hunyuan_nodes()
        i2v_encode = nodes["TextEncodeHunyuanVideo_ImageToVideo"]()
        empty_latent = nodes["EmptyHunyuanLatentVideo"]()
        text_encode = nodes["CLIPTextEncode"]()

        shots = ordered_shots(timeline)
        images = timeline.get("_runtime", {}).get("images")
        g = timeline.get("global", {})
        width, height = g.get("width", 960), g.get("height", 544)

        results = []
        for shot in shots:
            prompt, negative = resolve_prompt(timeline, shot)
            length = int(shot.get("duration_frames", 65))
            img_tensor = _resolve_image_ref(images, shot.get("image_ref"))

            (neg_cond,) = text_encode.encode(clip=clip, text=negative or "")

            if img_tensor is not None:
                # image_interleave controls how strongly/often the image
                # conditioning re-asserts itself through the sequence;
                # 1 = every frame, higher = sparser. Default to a value
                # that favors prompt adherence while still respecting the
                # source image, similar to LTX's default guide strength.
                (pos_cond,) = i2v_encode.encode(
                    clip=clip,
                    clip_vision_output=clip_vision,
                    prompt=prompt,
                    image_interleave=2,
                )
                latent = _encode_image_latent(vae, img_tensor, length, width, height)
            else:
                (pos_cond,) = text_encode.encode(clip=clip, text=prompt)
                (latent,) = empty_latent.generate(width=width, height=height, length=length, batch_size=1)

            results.append({
                "shot_id": shot["id"],
                "positive": pos_cond,
                "negative": neg_cond,
                "latent": latent,
                "num_frames": length,
            })

        return (results, len(results))


def _resolve_image_ref(images, ref):
    if images is None or ref is None:
        return None
    try:
        return images[ref:ref + 1]
    except (IndexError, TypeError):
        return None


def _encode_image_latent(vae, img_tensor, length, width, height):
    import torch
    import torch.nn.functional as F

    resized = F.interpolate(
        img_tensor.movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False
    ).movedim(1, -1)
    encoded = vae.encode(resized[:, :, :, :3])
    # Pad/replicate to the requested frame count along the temporal axis
    # so downstream samplers get a latent of the expected shape even
    # though we only had one real reference frame to encode.
    if encoded.dim() == 5:
        t = encoded.shape[2]
        target_t = max(1, length // 4 + 1)
        if t < target_t:
            pad = encoded[:, :, -1:].repeat(1, 1, target_t - t, 1, 1)
            encoded = torch.cat([encoded, pad], dim=2)
    return {"samples": encoded}


NODE_CLASS_MAPPINGS = {
    "TVToHunyuanAdapter": TVToHunyuanAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TVToHunyuanAdapter": "TV → Hunyuan Adapter 🎬",
}
