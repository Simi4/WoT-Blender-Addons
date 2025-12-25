


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
# decorators

def disable_undo(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        use_global_undo_orig = user_preferences.edit.use_global_undo
        user_preferences.edit.use_global_undo = False
        func(*args, **kwargs)
        user_preferences.edit.use_global_undo = use_global_undo_orig
    return wrapper


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
    if 'true' in bool_str.lower():
        return True
    return False


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


def tv_UnpackNormal(packed):
    pky = (packed>>22)&0x1FF
    pkz = (packed>>11)&0x3FF
    pkx = packed&0x3FF
    x = pkx/1023.0
    if pkx & (1<<10):
        x = -x
    y = pky/511.0
    if pky & (1<<9):
        y = -y
    z = pkz/1023.0
    if pkz & (1<<10):
        z = -z
    return Vector((x, z, y))


def tv_UnpackNormal_tag3(packed):
    pkz = (packed>>16)&0xFF^0xFF
    pky = (packed>>8)&0xFF^0xFF
    pkx = packed&0xFF^0xFF
    if pkx > 0x7f:
        x = -float(pkx&0x7f)/0x7f
    else:
        x = float(pkx^0x7f)/0x7f
    if pky > 0x7f:
        y = -float(pky&0x7f)/0x7f
    else:
        y = float(pky^0x7f)/0x7f
    if pkz>0x7f:
        z = -float(pkz&0x7f)/0x7f
    else:
        z = float(pkz^0x7f)/0x7f
    return Vector((x, z, y))
