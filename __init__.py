"""
ComfyUI-BAT-CorridorKey2 -- CorridorKey 2.0 green/blue-screen keying in ComfyUI.

CorridorKey 2.0 is by Corridor Digital (model designed and trained by Zachary
Harris with Niko Pueringer): https://github.com/nikopueringer/CorridorKey2.0

This pack contains none of their code. It downloads the pinned upstream release
on first use and drives it from ComfyUI tensors. See README.md and LICENSE.
"""

from .ck2.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
