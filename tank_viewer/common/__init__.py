


#####################################################################
# imports

import logging
import functools
from pathlib import Path
from typing import Optional
from ..common.XmlUnpacker import XmlUnpacker
from ..settings import ADDON_PREF_ID

# blender imports
import bpy
from mathutils import Vector, Matrix


#####################################################################
# globals

logging.basicConfig()
logger = logging.getLogger('tank_viewer')
logger.setLevel(logging.INFO)

user_preferences = bpy.context.preferences

g_XmlUnpacker = XmlUnpacker()


#####################################################################
# functions

def load_image_from_memory(data: bytes, name: str) -> bpy.types.Image:
    image = bpy.data.images.new(name, 8, 8)
    image.pack(data=data, data_len=len(data))
    image.source = 'FILE'
    return image


def addon_prefs():
    return user_preferences.addons[ADDON_PREF_ID].preferences


def StrToVector(vector_str):
    if vector_str:
        return Vector(tuple(map(float, vector_str.strip().split())))


def tv_AsMatrix4x4T(vector_str):
    vector_16 = StrToVector(vector_str)
    return Matrix(
        (vector_16[:4], vector_16[4:8], vector_16[8:12], vector_16[12:16])).transposed()


def tv_AsBool(bool_str):
    return 'true' in bool_str.lower()


def AsInt(int_str):
    if int_str is None:
        logger.error('int_str is None')
    int_str = int_str.strip()
    if int_str.isdigit():
        return int(int_str)
    return 0


def StrToFilePath(path_str: Optional[str]) -> Path:
    if path_str is not None:
        return '/'.join(path_str.strip().split('\\'))
tv_AsNormPath = StrToFilePath
