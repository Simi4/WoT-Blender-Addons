from ctypes import c_float, c_uint32, c_int32
from .._base_json_section import *
from .v1_16_1 import BSMO_Section_1_16_1


class NodeItem_v1_46_0_RU(CStructure):
    _size_ = 72

    _fields_ = [
        ('transform', c_float * 16),
        ('parent_index', c_int32),
        ('identifier_fnv', c_uint32),
    ]


class BSMO_Section_1_46_0_RU(BSMO_Section_1_16_1):
    _fields_ = list(BSMO_Section_1_16_1._fields_)
    _fields_[11] = (list, 'visual_nodes', NodeItem_v1_46_0_RU)
