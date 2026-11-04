from struct import calcsize, unpack, pack
from .._base_json_section import *
from .v1_32_0_ru import BSMI_Section_1_32_0_RU


class BSMI_Section_1_46_0_RU(BSMI_Section_1_32_0_RU):
    @row_seek(True)
    def from_bin_stream(self, stream, row):
        self._data = {}
        for prop, name, data_type in self._fields_[:11]:
            if prop == list:
                self.read_vector(stream, name, data_type)
            elif prop == dict:
                self.read_dict(stream, name, data_type)
            elif prop == int:
                sz = calcsize(data_type)
                self._data[name] = unpack(data_type, stream.read(sz))[0]
        rem = row.position + row.length - stream.tell()
        if rem > 0:
            self._data['_tail_hex'] = stream.read(rem).hex()

    def to_bin(self):
        res = b''
        for prop, name, data_type in self._fields_[:11]:
            if prop == list:
                res += self.write_vector(name, data_type)
            elif prop == dict:
                res += self.write_dict(name, data_type)
            elif prop == int:
                res += pack(data_type, self._data[name])
        if '_tail_hex' in self._data:
            res += bytes.fromhex(self._data['_tail_hex'])
        return res
