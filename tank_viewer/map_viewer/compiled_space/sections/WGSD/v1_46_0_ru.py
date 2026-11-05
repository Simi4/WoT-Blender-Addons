from struct import unpack, pack
from ctypes import c_float, c_uint8, c_uint32
from .._base_json_section import *
from .v0_9_12 import Decal2Info
from .v1_0_0 import DecalInfo_TYPE1_v1_0_0


class DecalInfo_TYPE3_v1_46_0_RU(CStructure):
    _size_ = 136
    _pack_ = 1

    _fields_ = [
        ('_1', c_uint32),
        ('_2', c_uint32),
        ('accurate', c_uint8),
        ('transform', c_float * 16),
        ('diff_tex_fnv', c_uint32),
        ('bump_tex_fnv', c_uint32),
        ('hm_tex_fnv', c_uint32),
        ('add_tex_fnv', c_uint32),
        ('priority', c_uint32),
        ('influence', c_uint8),
        ('_3', c_uint8),
        ('offsets', c_float * 4),
        ('uv_wrapping', c_float * 2),
        ('visibility_mask', c_uint32),
        ('tiles_fade', c_float),
        ('unknown_4', c_uint8),
        ('parallax_offset', c_float),
        ('parallax_amplitude', c_float),
    ]


class WGSD_Section_1_46_0_RU(Base_JSON_Section):
    header = 'WGSD'
    int1 = 3

    @row_seek(True)
    def from_bin_stream(self, stream, row):
        self._data = {}

        int1 = unpack('<I', stream.read(4))[0]

        self._data['1_135'] = None
        self._data['1_127'] = None
        self._data['2_127'] = None
        self._data['2_135'] = None

        if int1 != 0:
            int2, int3, int4 = unpack('<3I', stream.read(12))

            if int2 == 3:
                self._data['1_135'] = []
                for i in range(int3):
                    self._data['1_135'].append(
                        DecalInfo_TYPE3_v1_46_0_RU(
                            stream.read(int4)
                        ).to_dict()
                    )

            elif int2 == 1:
                self._data['1_127'] = []
                for i in range(int3):
                    self._data['1_127'].append(
                        DecalInfo_TYPE1_v1_0_0(
                            stream.read(int4)
                        ).to_dict()
                    )

            if int1 == 2:
                int5, int6, int7 = unpack('<3I', stream.read(12))

                if int5 == 1:
                    self._data['2_127'] = []
                    for i in range(int6):
                        self._data['2_127'].append(
                            DecalInfo_TYPE1_v1_0_0(
                                stream.read(int7)
                            ).to_dict()
                        )

                elif int5 == 3:
                    self._data['2_135'] = []
                    for i in range(int6):
                        self._data['2_135'].append(
                            DecalInfo_TYPE3_v1_46_0_RU(
                                stream.read(int7)
                            ).to_dict()
                        )

        self._data['3'] = self.read_entries(stream, Decal2Info)

    def to_bin(self):
        res = b''

        _1_135 = self._data['1_135']
        _1_127 = self._data['1_127']
        _2_127 = self._data['2_127']
        _2_135 = self._data['2_135']

        if _2_127 or _2_135:
            res += pack('<I', 2)
        elif _1_135 or _1_127:
            res += pack('<I', 1)
        else:
            res += pack('<I', 0)

        if _1_135:
            cnt = len(_1_135)
            res += pack('<3I', 3, cnt, 136)

            for it in _1_135:
                res += DecalInfo_TYPE3_v1_46_0_RU(it).to_bin()

        elif _1_127:
            cnt = len(_1_127)
            res += pack('<3I', 1, cnt, 128)

            for it in _1_127:
                res += DecalInfo_TYPE1_v1_0_0(it).to_bin()

        if _2_135:
            cnt = len(_2_135)
            res += pack('<3I', 3, cnt, 136)

            for it in _2_135:
                res += DecalInfo_TYPE3_v1_46_0_RU(it).to_bin()

        if _2_127:
            cnt = len(_2_127)
            res += pack('<3I', 1, cnt, 128)

            for it in _2_127:
                res += DecalInfo_TYPE1_v1_0_0(it).to_bin()

        res += self.write_entries(self._data['3'], Decal2Info)

        return res
