"""
TensorVizion Director — model-agnostic timeline node + per-backend adapters
for LTX, Wan, and Hunyuan video conditioning.

Follows the same conventions as TensorVizion/OmniNodes:
  - per-file NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
  - emoji display names
  - CATEGORY = "TensorVizion/Director/..."
  - auto-discovery so new adapters just need to define the two mapping
    dicts and drop into adapters/ — no manual registration required here
"""

import importlib
import pkgutil
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./js"

_MODULES = [
    "director_node",
]

_ADAPTER_PACKAGE = "adapters"


def _merge_mappings(module):
    cls_map = getattr(module, "NODE_CLASS_MAPPINGS", None)
    name_map = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None)
    if cls_map:
        NODE_CLASS_MAPPINGS.update(cls_map)
    if name_map:
        NODE_DISPLAY_NAME_MAPPINGS.update(name_map)


for mod_name in _MODULES:
    try:
        module = importlib.import_module(f".{mod_name}", package=__name__)
        _merge_mappings(module)
    except Exception:
        print(f"[TensorVizion Director] Failed to load {mod_name}:")
        traceback.print_exc()

try:
    adapters_pkg = importlib.import_module(f".{_ADAPTER_PACKAGE}", package=__name__)
    for _, mod_name, _ in pkgutil.iter_modules(adapters_pkg.__path__):
        try:
            module = importlib.import_module(f".{_ADAPTER_PACKAGE}.{mod_name}", package=__name__)
            _merge_mappings(module)
        except Exception:
            print(f"[TensorVizion Director] Failed to load adapter {mod_name}:")
            traceback.print_exc()
except Exception:
    print("[TensorVizion Director] Failed to load adapters package:")
    traceback.print_exc()

print(f"[TensorVizion Director] Loaded {len(NODE_CLASS_MAPPINGS)} node(s): "
      f"{', '.join(NODE_CLASS_MAPPINGS.keys())}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
