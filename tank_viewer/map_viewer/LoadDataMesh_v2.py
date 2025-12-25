
# imports
from dataclasses import dataclass
from struct import unpack

# blender imports
import numpy as np

# local imports
from ..common import *
from ..vertex_types import *


@dataclass
class PrimitiveGroup:
    startIndex: int
    nPrimitives: int
    startVertex: int
    nVertices: int


class LoadDataMesh_v2:
    is_processed = False
    PrimitiveGroups: list[PrimitiveGroup] = None
    uv2_list = None
    uv_list = None
    vertices = None
    indices = None
    __PackedGroups = None
    __pfile = None

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

    def __init__(self, file, vertices_name: str, primitive_name: str):
        file.seek(0)
        self.__pfile = file

        header = unpack('<I', self.__pfile.read(4))[0]
        assert header == 0x42A14E65

        self.__load_packed_section()

        self.__load_XYZNUV(
            self.__PackedGroups[primitive_name]['position'],
            self.__PackedGroups[vertices_name]['position'],
        )

        if (uv2_name := vertices_name[:-8] + 'uv2') in self.__PackedGroups:
            self.__load_uv2(
                self.__PackedGroups[uv2_name]['position'],
                self.__PackedGroups[uv2_name]['length'],
            )

    def __load_packed_section(self):
        self.__pfile.seek(-4, 2)
        table_start = unpack('<l', self.__pfile.read(4))[0]
        self.__pfile.seek(-4 - table_start, 2)
        position = 4
        self.__PackedGroups = {}

        while True:
            data = self.__pfile.read(4)
            if data is None or len(data) != 4:
                break

            section_size = unpack('<I', data)[0]
            data = self.__pfile.read(16)
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

    def __load_XYZNUV(self, iposition, vposition):
        self.__pfile.seek(iposition)
        indexFormat = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')
        nIndices = unpack('<I', self.__pfile.read(4))[0]
        nTriangleGroups = unpack('<H', self.__pfile.read(2))[0]
        self.PrimitiveGroups = []

        if indexFormat == 'list':
            UINT_LEN = 2

        elif indexFormat == 'list32':
            UINT_LEN = 4

        else:
            raise Exception(f'{indexFormat=}')

        offset = nIndices * UINT_LEN + 72

        self.__pfile.seek(iposition + offset)
        for i in range(nTriangleGroups):
            startIndex, nPrimitives, startVertex, nVertices = unpack('<4I', self.__pfile.read(16))
            self.PrimitiveGroups.append(PrimitiveGroup(startIndex, nPrimitives, startVertex, nVertices))

        self.__pfile.seek(vposition)
        vertices_subname = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')
        vertexFormat = ''
        if 'BPVT' in vertices_subname:
            self.is_processed = True
            self.__pfile.read(4)
            vertexFormat = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')

        verticesCount = unpack('<l', self.__pfile.read(4))[0]

        match vertexFormat:
            case vt_SET3_XYZNUVIIIWWTBPC.V_TYPE:
                vt = vt_SET3_XYZNUVIIIWWTBPC

            case vt_SET3_XYZNUVTBPC.V_TYPE:
                vt = vt_SET3_XYZNUVTBPC

            case vt_SET3_XYZNUVPC.V_TYPE:
                vt = vt_SET3_XYZNUVPC

            case _:
                if vt_XYZNUVIIIWWTB.V_TYPE in vertices_subname:
                    vt = vt_XYZNUVIIIWWTB

                elif vt_XYZNUVTB.V_TYPE in vertices_subname:
                    vt = vt_XYZNUVTB

                elif vt_XYZNUV.V_TYPE in vertices_subname:
                    vt = vt_XYZNUV

                else:
                    logger.warning(f'{vertexFormat=}; {vertices_subname=}')
                    return

        vertices_list = np.frombuffer(self.__pfile.read(verticesCount * vt.SIZE), vt.DTYPE)

        self.vertices = vertices_list['pos'].copy()
        self.uv_list = vertices_list['uv'].copy()

        # FIXME: move to shader ?
        self.uv_list[:, 1] = 1.0 - self.uv_list[:, 1]

        self.__pfile.seek(iposition + 72)
        self.indices = np.frombuffer(
            self.__pfile.read(nIndices * UINT_LEN),
            ('<u4' if UINT_LEN == 4 else '<u2', 3),
        ).copy()

        # swap 0 <-> 2
        self.indices[:, 0], self.indices[:, 2] = (
            self.indices[:, 2],
            self.indices[:, 0].copy(),
        )

    def __load_uv2(self, uv2_position, uv2_length):
        self.__pfile.seek(uv2_position)

        if not self.is_processed:
            self.uv2_list = np.frombuffer(self.__pfile.read(uv2_length), ('<f4', 2)).copy()

            # FIXME: move to shader ?
            self.uv2_list[:, 1] = 1.0 - self.uv2_list[:, 1]

        else:
            data = self.__pfile.read(64)
            uv2_subname = data.split(b'\x00')[0].decode('utf-8')

            uv2_format = ''
            if 'BPVS' in uv2_subname:
                self.__pfile.read(4)
                uv2_format = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')

            uv2_Count = unpack('<I', self.__pfile.read(4))[0]

            if uv2_format == 'set3/uv2pc':
                self.uv2_list = np.frombuffer(self.__pfile.read(uv2_Count * 8), ('<f4', 2)).copy()

                # FIXME: move to shader ?
                self.uv2_list[:, 1] = 1.0 - self.uv2_list[:, 1]

            else:
                logger.warning(f'{uv2_format=}')
