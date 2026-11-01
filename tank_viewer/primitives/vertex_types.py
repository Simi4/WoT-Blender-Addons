# original: https://github.com/konlil/mypy/blob/master/vertices.py

from enum import Enum, auto
from typing import ClassVar


class NormalFormat(Enum):
    UNPACKED = auto()
    PACKED_11_11_10 = auto()
    PACKED_8_8_8 = auto()


class vt_baseType:
    SIZE: ClassVar[int]
    IS_SKINNED: ClassVar[bool]
    NORMAL_FMT = ClassVar[NormalFormat]
    V_TYPE: ClassVar[bytes]
    DTYPE: ClassVar[list]


class vt_SET3_XYZNUVTBPC(vt_baseType):
    SIZE = 32
    IS_SKINNED = False
    NORMAL_FMT = NormalFormat.PACKED_8_8_8
    V_TYPE = b"set3/xyznuvtbpc"
    DTYPE = [
        ("pos", "<f4", 3),
        ("packed_normal", "<i4"),
        ("uv", "<f4", 2),
        ("packed_tb", "<i4", 2),
    ]


class vt_SET3_XYZNUVPC(vt_baseType):
    SIZE = 24
    IS_SKINNED = False
    NORMAL_FMT = NormalFormat.PACKED_8_8_8
    V_TYPE = b"set3/xyznuvpc"
    DTYPE = [
        ("pos", "<f4", 3),
        ("packed_normal", "<i4"),
        ("uv", "<f4", 2),
    ]


class vt_SET3_XYZNUVIIIWWTBPC(vt_baseType):
    SIZE = 40
    IS_SKINNED = True
    NORMAL_FMT = NormalFormat.PACKED_8_8_8
    V_TYPE = b"set3/xyznuviiiwwtbpc"
    DTYPE = [
        ("pos", "<f4", 3),
        ("packed_normal", "<u4"),
        ("uv", "<f4", 2),
        ("iiiww", "B", 8),
        ("packed_tb", "<u4", 2),
    ]


class vt_XYZNUVIIIWWTB(vt_baseType):
    SIZE = 37
    IS_SKINNED = True
    NORMAL_FMT = NormalFormat.PACKED_11_11_10
    V_TYPE = b"xyznuviiiwwtb"
    DTYPE = [
        ("pos", "<f4", 3),
        ("packed_normal", "<u4"),
        ("uv", "<f4", 2),
        ("iiiww", "B", 5),
        ("packed_tb", "<u4", 2),
    ]


class vt_XYZNUVTB(vt_baseType):
    SIZE = 32
    IS_SKINNED = False
    NORMAL_FMT = NormalFormat.PACKED_11_11_10
    V_TYPE = b"xyznuvtb"
    DTYPE = [
        ("pos", "<f4", 3),
        ("packed_normal", "<u4"),
        ("uv", "<f4", 2),
        ("packed_tb", "<u4", 2),
    ]


class vt_XYZNUV(vt_baseType):
    SIZE = 32
    IS_SKINNED = False
    NORMAL_FMT = NormalFormat.UNPACKED
    V_TYPE = b"xyznuv"
    DTYPE = [
        ("pos", "<f4", 3),
        ("normal", "<f4", 3),
        ("uv", "<f4", 2),
    ]
