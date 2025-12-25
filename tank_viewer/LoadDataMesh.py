



#####################################################################
# imports

from .common import *
from .vertex_types import *

from struct import unpack



#####################################################################
# LoadDataMesh

class LoadDataMesh:
    PrimitiveGroups = None
    uv2_list = None
    uv_list = None
    # normal_list = None
    # tangent_list = None
    # binormal_list = None
    vertices = None
    indices = None
    __uv2_list = None
    __PackedGroups = None
    __pfile = None

    def __init__(self, file,
                 vertices_name='vertices',
                 primitive_name='indices',
                 uv2_name='', new_ext_format=True,
                 colour_name=''):

        file.seek(0)
        self.__pfile = file

        header = unpack('<I', self.__pfile.read(4))[0]
        assert(header == 0x42a14e65)

        self.__load_packed_section()

        if uv2_name:
            if self.__PackedGroups.get(uv2_name):
                self.__load_uv2(
                    self.__PackedGroups[uv2_name]['position'],
                    self.__PackedGroups[uv2_name]['length'],
                    new_ext_format
                )

        self.__load_XYZNUV(
            self.__PackedGroups[primitive_name]['position'],
            self.__PackedGroups[vertices_name]['position'])


    def __load_packed_section(self):
        self.__pfile.seek(-4, 2)
        table_start = unpack('<l', self.__pfile.read(4))[0]
        self.__pfile.seek(-4-table_start, 2)
        position = 4
        self.__PackedGroups = {}

        while True:
            data = self.__pfile.read(4)
            if data == None or len(data) != 4:
                break

            section_size = unpack('<I', data)[0]
            data = self.__pfile.read(16)
            data = self.__pfile.read(4)
            if data == None or len(data) != 4:
                break

            section_name_length = unpack('<I', data)[0]
            section_name = self.__pfile.read(section_name_length).decode('utf-8')
            for item in ('vertices', 'indices', 'uv2', 'colour'):
                if item in section_name:
                    self.__PackedGroups[section_name] = {
                        'position' : position,
                        'length' : section_size
                    }
                    break

            position += section_size

            if section_size%4 > 0:
                position += 4-section_size%4

            if section_name_length%4 > 0:
                self.__pfile.read(4-section_name_length%4)


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

        offset = nIndices*UINT_LEN+72

        self.__pfile.seek(iposition+offset)
        for i in range(nTriangleGroups):
            startIndex = unpack('<I', self.__pfile.read(4))[0]
            nPrimitives = unpack('<I', self.__pfile.read(4))[0]
            startVertex = unpack('<I', self.__pfile.read(4))[0]
            nVertices = unpack('<I', self.__pfile.read(4))[0]
            self.PrimitiveGroups.append({
                'startIndex' : startIndex,
                'nPrimitives' : nPrimitives,
                'startVertex' : startVertex,
                'nVertices' : nVertices
            })

        self.__pfile.seek(vposition)
        vertices_subname = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')
        vertexFormat = ''
        if 'BPVT' in vertices_subname:
            self.__pfile.read(4)
            vertexFormat = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')

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

        verticesCount = unpack('<l', self.__pfile.read(4))[0]

        vertices_list = []
        for _ in range(verticesCount):
            vertex = vt(unpack(vt.FORMAT, self.__pfile.read(vt.SIZE)))
            vertices_list.append(vertex)

        old2new = {}
        new_vertices_list = {}
        vidx = 0

        for i, vert in enumerate(vertices_list):
            if self.__uv2_list:
                vert.uv2 = self.__uv2_list[i]

            new_vert = vert.to_tuple()

            if new_vert not in new_vertices_list:
                old2new[i] = vidx
                new_vertices_list[new_vert] = vidx
                vidx += 1

            else:
                old2new[i] = new_vertices_list[new_vert]

        del vertices_list

        new_vertices_list = dict((v, k) for k, v in new_vertices_list.items())
        new_vertices_list = list(new_vertices_list.values())

        if not vertexFormat in (vt_SET3_XYZNUVPC.V_TYPE, vt_XYZNUV.V_TYPE):
            if self.__uv2_list:
                del self.__uv2_list
                (self.vertices, _, self.uv_list, self.uv2_list, _, _) = zip(*new_vertices_list)

            else:
                (self.vertices, _, self.uv_list, _, _) = zip(*new_vertices_list)

        else:
            if self.__uv2_list:
                del self.__uv2_list
                (self.vertices, _, self.uv_list, self.uv2_list) = zip(*new_vertices_list)

            else:
                (self.vertices, _, self.uv_list) = zip(*new_vertices_list)

        self.indices = []
        for group in self.PrimitiveGroups:
            self.__pfile.seek(iposition + group['startIndex']*UINT_LEN+72)
            for cnt in range(group['nPrimitives']):
                if UINT_LEN == 2:
                    v1 = unpack('<H', self.__pfile.read(2))[0]
                    v2 = unpack('<H', self.__pfile.read(2))[0]
                    v3 = unpack('<H', self.__pfile.read(2))[0]

                elif UINT_LEN == 4:
                    v1 = unpack('<I', self.__pfile.read(4))[0]
                    v2 = unpack('<I', self.__pfile.read(4))[0]
                    v3 = unpack('<I', self.__pfile.read(4))[0]

                self.indices.append((old2new[v3], old2new[v2], old2new[v1]))


    def __load_uv2(self, uv2_position, uv2_length, new_ext_format):
        self.__pfile.seek(uv2_position)

        if new_ext_format:
            uv2_subname = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')

            uv2_format = ''
            if 'BPVS' in uv2_subname:
                self.__pfile.read(4)
                uv2_format = self.__pfile.read(64).split(b'\x00')[0].decode('utf-8')

            uv2_Count = unpack('<I', self.__pfile.read(4))[0]

            if uv2_format == 'set3/uv2pc':
                self.__uv2_list = []
                for i in range(uv2_Count):
                    (u, v) = unpack('<2f', self.__pfile.read(8))
                    self.__uv2_list.append((u, 1-v))

            else:
                logger.warning(f'{uv2_format=}')

        else:
            uv2_Count = uv2_length//8
            self.__uv2_list = []
            for i in range(uv2_Count):
                (u, v) = unpack('<2f', self.__pfile.read(8))
                self.__uv2_list.append((u, 1-v))
