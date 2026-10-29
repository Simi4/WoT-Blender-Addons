__all__ = ('LoadDataMesh', 'PrimitiveGroup')


from dataclasses import dataclass
from struct import unpack

import numpy as np

from .common import logger
from .vertex_types import (
    vt_SET3_XYZNUVIIIWWTBPC,
    vt_SET3_XYZNUVTBPC,
    vt_SET3_XYZNUVPC,
    vt_XYZNUV,
    vt_XYZNUVIIIWWTB,
    vt_XYZNUVTB,
)


@dataclass
class PrimitiveGroup:
    startIndex: int
    nPrimitives: int
    startVertex: int
    nVertices: int


class LoadDataMesh:
    def __init__(
        self,
        file,
        vertices_name='vertices',
        primitive_name='indices',
        uv2_name='',
        new_ext_format=True,
        colour_name='',
        *,
        deduplicate=True,
        transform_coordinates=True,
    ):
        file.seek(0)
        self.__pfile = file
        self.is_processed = False
        self.PrimitiveGroups = []
        self.vertices = None
        self.indices = None
        self.uv_list = None
        self.uv2_list = None
        self.__PackedGroups = {}
        self.__deduplicate = deduplicate
        self.__transform_coordinates = transform_coordinates

        header = unpack('<I', self.__pfile.read(4))[0]
        assert header == 0x42A14E65

        self.__load_packed_section()

        if uv2_name is None:
            candidate = vertices_name[:-8] + 'uv2'
            uv2_name = candidate if candidate in self.__PackedGroups else ''

        self.__load_XYZNUV(
            self.__PackedGroups[primitive_name]['position'],
            self.__PackedGroups[vertices_name]['position'],
        )

        if uv2_name and uv2_name in self.__PackedGroups:
            self.__load_uv2(
                self.__PackedGroups[uv2_name]['position'],
                self.__PackedGroups[uv2_name]['length'],
            )

        if self.__deduplicate:
            self.__deduplicate_vertices()

    def get_vertices_by_id(self, idx: int) -> np.ndarray:
        pg = self.PrimitiveGroups[idx]
        return self.vertices[pg.startVertex : pg.startVertex + pg.nVertices]

    def get_indices_by_id(self, idx: int) -> np.ndarray:
        pg = self.PrimitiveGroups[idx]
        indices = self.indices[pg.startIndex // 3 : pg.startIndex // 3 + pg.nPrimitives]
        return indices - pg.startVertex

    def get_uv_by_id(self, idx: int) -> np.ndarray:
        pg = self.PrimitiveGroups[idx]
        return self.uv_list[pg.startVertex : pg.startVertex + pg.nVertices]

    def get_uv2_by_id(self, idx: int) -> np.ndarray:
        pg = self.PrimitiveGroups[idx]
        return self.uv2_list[pg.startVertex : pg.startVertex + pg.nVertices]

    def __load_packed_section(self):
        self.__pfile.seek(-4, 2)
        table_start = unpack('<l', self.__pfile.read(4))[0]
        self.__pfile.seek(-4 - table_start, 2)
        position = 4

        while True:
            data = self.__pfile.read(4)
            if data is None or len(data) != 4:
                break

            section_size = unpack('<I', data)[0]
            self.__pfile.read(16)
            data = self.__pfile.read(4)
            if data is None or len(data) != 4:
                break

            section_name_length = unpack('<I', data)[0]
            section_name = self.__pfile.read(section_name_length).decode('utf-8')
            for item in ('vertices', 'indices', 'uv2', 'colour'):
                if item in section_name:
                    self.__PackedGroups[section_name] = {
                        'position': position,
                        'length': section_size,
                    }
                    break

            position += section_size
            if section_size % 4 > 0:
                position += 4 - section_size % 4

            if section_name_length % 4 > 0:
                self.__pfile.read(4 - section_name_length % 4)

    @staticmethod
    def __get_vertex_type(vertices_subname, vertexFormat):
        match vertexFormat:
            case vt_SET3_XYZNUVIIIWWTBPC.V_TYPE:
                return vt_SET3_XYZNUVIIIWWTBPC
            case vt_SET3_XYZNUVTBPC.V_TYPE:
                return vt_SET3_XYZNUVTBPC
            case vt_SET3_XYZNUVPC.V_TYPE:
                return vt_SET3_XYZNUVPC

        if vt_XYZNUVIIIWWTB.V_TYPE in vertices_subname:
            return vt_XYZNUVIIIWWTB
        if vt_XYZNUVTB.V_TYPE in vertices_subname:
            return vt_XYZNUVTB
        if vt_XYZNUV.V_TYPE in vertices_subname:
            return vt_XYZNUV
        return None

    def __load_XYZNUV(self, iposition, vposition):
        self.__pfile.seek(iposition)
        indexFormat = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')
        nIndices = unpack('<I', self.__pfile.read(4))[0]
        nTriangleGroups = unpack('<H', self.__pfile.read(2))[0]

        if indexFormat == 'list':
            index_dtype = '<u2'
        elif indexFormat == 'list32':
            index_dtype = '<u4'
        else:
            raise ValueError(f'Unsupported index format: {indexFormat}')

        self.__pfile.seek(iposition + nIndices * np.dtype(index_dtype).itemsize + 72)
        for _ in range(nTriangleGroups):
            values = unpack('<4I', self.__pfile.read(16))
            self.PrimitiveGroups.append(PrimitiveGroup(*values))

        self.__pfile.seek(vposition)
        vertices_subname = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')
        vertexFormat = ''
        if 'BPVT' in vertices_subname:
            self.is_processed = True
            self.__pfile.read(4)
            vertexFormat = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')

        verticesCount = unpack('<l', self.__pfile.read(4))[0]
        vt = self.__get_vertex_type(vertices_subname, vertexFormat)
        if vt is None:
            logger.warning(f'{vertexFormat=}; {vertices_subname=}')
            return

        vertices_data = np.frombuffer(
            self.__pfile.read(verticesCount * vt.SIZE),
            dtype=vt.DTYPE,
            count=verticesCount,
        )
        if self.__deduplicate:
            self.__source_vertices = vertices_data

        self.vertices = vertices_data['pos'].copy()
        self.uv_list = vertices_data['uv'].copy()
        self.uv_list[:, 1] = 1.0 - self.uv_list[:, 1]

        if self.__transform_coordinates:
            if vt.IS_SKINNED:
                self.vertices[:, 1] = -vertices_data['pos'][:, 2]
            else:
                self.vertices[:, 1] = vertices_data['pos'][:, 2]
            self.vertices[:, 2] = vertices_data['pos'][:, 1]

        self.__pfile.seek(iposition + 72)
        indices = np.frombuffer(
            self.__pfile.read(nIndices * np.dtype(index_dtype).itemsize),
            dtype=np.dtype((index_dtype, 3)),
        )
        self.indices = indices[:, [2, 1, 0]].copy()

    def __load_uv2(self, uv2_position, uv2_length):
        self.__pfile.seek(uv2_position)

        if not self.is_processed:
            self.uv2_list = np.frombuffer(
                self.__pfile.read(uv2_length),
                dtype=('<f4', 2),
            ).copy()
        else:
            uv2_subname = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')
            uv2_format = ''
            if 'BPVS' in uv2_subname:
                self.__pfile.read(4)
                uv2_format = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')

            uv2_Count = unpack('<I', self.__pfile.read(4))[0]
            if uv2_format != 'set3/uv2pc':
                logger.warning(f'{uv2_format=}')
                return

            self.uv2_list = np.frombuffer(
                self.__pfile.read(uv2_Count * 8),
                dtype=('<f4', 2),
                count=uv2_Count,
            ).copy()

        self.uv2_list[:, 1] = 1.0 - self.uv2_list[:, 1]

    def __deduplicate_vertices(self):
        if self.vertices is None or len(self.vertices) == 0:
            return

        source = self.__source_vertices
        normal_name = 'packed_normal' if 'packed_normal' in source.dtype.names else 'normal'
        key_dtype = [
            ('pos', self.vertices.dtype, (3,)),
            (normal_name, source.dtype[normal_name]),
            ('uv', self.uv_list.dtype, (2,)),
        ]
        if self.uv2_list is not None:
            key_dtype.append(('uv2', self.uv2_list.dtype, (2,)))
        if 'packed_tb' in source.dtype.names:
            key_dtype.append(('packed_tb', source.dtype['packed_tb']))

        keys = np.empty(len(source), dtype=key_dtype)
        keys['pos'] = self.vertices
        keys[normal_name] = source[normal_name]
        keys['uv'] = self.uv_list
        if self.uv2_list is not None:
            keys['uv2'] = self.uv2_list
        if 'packed_tb' in source.dtype.names:
            keys['packed_tb'] = source['packed_tb']

        _, first_indices, inverse = np.unique(
            keys,
            return_index=True,
            return_inverse=True,
        )
        order = np.argsort(first_indices)
        selected = first_indices[order]

        old_to_new = np.empty(len(order), dtype=np.int32)
        old_to_new[order] = np.arange(len(order), dtype=np.int32)
        inverse = inverse.reshape(-1)
        self.indices = old_to_new[inverse][self.indices]

        self.vertices = self.vertices[selected]
        self.uv_list = self.uv_list[selected]
        if self.uv2_list is not None:
            self.uv2_list = self.uv2_list[selected]

        del self.__source_vertices
