from struct import calcsize
from .._base_json_section import *
from .v1_32_0_ru import BSMI_Section_1_32_0_RU


class BSMI_Section_1_46_0_RU(BSMI_Section_1_32_0_RU):
    @row_seek(True)
    def from_bin_stream(self, stream, row):
        stream_name = str(getattr(stream, 'name', ''))
        if '503_kaliningrad' in stream_name:
            self._data = {}
            for prop, name, data_type in self._fields_[:11]:
                if prop == list:
                    self.read_vector(stream, name, data_type)
                elif prop == dict:
                    self.read_dict(stream, name, data_type)
                elif prop == int:
                    sz = calcsize(data_type)
                    self._data[name] = my_unpack(data_type, stream.read(sz))
            self._data['13_4'] = 0
            self._data['14_4'] = []
            self._data['15_4'] = 0
            self._data['16_4'] = []
            stream.seek(row.position + row.length)
        else:
            self._data = {}
            for prop, name, data_type in self._fields_:
                if prop == list:
                    self.read_vector(stream, name, data_type)
                elif prop == dict:
                    self.read_dict(stream, name, data_type)
                elif prop == int:
                    sz = calcsize(data_type)
                    self._data[name] = my_unpack(data_type, stream.read(sz))
