"""
TV Wan Shot Iterator.

WanImageToVideo / WanFirstLastFrameToVideo produce one conditioning+latent
triple per clip, and Wan doesn't support continuously injecting guides
across an arbitrary number of shots the way LTX does. So TVToWanAdapter
returns a list of per-shot bundles, and this node lets a workflow pull
one shot out by index to sample it individually. Wire a for-loop
(e.g. via ComfyUI's native list-execution or an Impact-Pack style loop)
around this node, sampling each shot's latent separately, then
concatenate/stitch the resulting videos downstream (e.g. with a video
combine node) in timeline order.
"""

from __future__ import annotations


class TVWanShotIterator:
    NODE_NAME = "TV Wan Shot Iterator 🎬"
    CATEGORY = "TensorVizion/Director/Adapters"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wan_shots": ("TV_WAN_SHOTLIST",),
                "index": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "num_frames", "shot_id")
    FUNCTION = "pick"
    CATEGORY = CATEGORY

    def pick(self, wan_shots, index):
        if not wan_shots:
            raise ValueError("TV Wan Shot Iterator received an empty shot list.")
        if index >= len(wan_shots):
            raise ValueError(
                f"Index {index} out of range — this timeline only has "
                f"{len(wan_shots)} Wan shot(s) (0..{len(wan_shots) - 1})."
            )
        shot = wan_shots[index]
        return (
            shot["positive"],
            shot["negative"],
            shot["latent"],
            shot["num_frames"],
            shot["shot_id"],
        )


NODE_CLASS_MAPPINGS = {
    "TVWanShotIterator": TVWanShotIterator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TVWanShotIterator": "TV Wan Shot Iterator 🎬",
}
