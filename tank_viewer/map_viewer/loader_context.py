
# imports
from dataclasses import dataclass, field

# blender imports
import bpy  # type: ignore

# local imports
from .compiled_space.universal_space import UniversalResMgr, UniversalSpace


@dataclass
class LoaderContext:
    res_mgr: UniversalResMgr
    space: UniversalSpace
    map_info: dict
    image_cache: dict[str, bpy.types.Image] = field(default_factory=dict)
