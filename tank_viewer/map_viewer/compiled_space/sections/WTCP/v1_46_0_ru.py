""" WTCP (WoT static scene Control Point) """
from ctypes import c_float, c_uint32
from .._base_json_section import *


class ControlPoint_v1_46_0_RU(CStructure):
    _size_ = 136
    _pack_ = 1

    _fields_ = [
        ('transform', c_float * 16),
        ('radius_color', c_float * 4),
        ('radius', c_float),
        ('over_terrain_height', c_float),
        ('flag_scale', c_float),
        ('capture_radius_sq', c_float),
        ('points_per_second', c_float),
        ('fade_speed', c_float),
        ('visibility_mask', c_uint32),
        ('team', c_uint32),
        ('base_id', c_uint32),
        ('flag_path_fnv', c_uint32),
        ('flagstaff_path_fnv', c_uint32),
        ('radius_path_fnv', c_uint32),
        ('wwevent_name_fnv', c_uint32),
        ('before_wind', c_uint32, 1),
        ('pad', c_uint32, 31),
    ]


class WTCP_Section_1_46_0_RU(Base_JSON_Section):
    header = 'WTCP'
    int1 = 3

    _fields_ = [
        (list, 'control_points', ControlPoint_v1_46_0_RU),
    ]
