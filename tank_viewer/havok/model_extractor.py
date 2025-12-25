"""wiiskii.g@gmail.com"""

import logging
from dataclasses import dataclass
from typing import IO, TypeAlias
import numpy as np

from .tag_tools import TagFileType, TagObject, TagReader

logger = logging.getLogger(__name__)

Vec3f: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]
Vec3fArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]
Quad4iArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.int64]]
Ui32Array: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint32]]
Ui64Array: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint64]]


def unpack_packed_vertices(
    packed: Ui32Array,
    codec: 'CodecParams',
    sec_domain_min: Vec3f,
    sec_domain_max: Vec3f,
) -> Vec3fArray:
    res = np.empty((packed.size, 3), dtype=np.float32)
    res[:, 2] = packed >> 22 & 0x3FF
    res[:, 1] = packed >> 11 & 0x7FF
    res[:, 0] = packed & 0x7FF

    res *= codec.scale_xyz
    res += codec.min_xyz
    np.clip(res, sec_domain_min, sec_domain_max, out=res)
    return res


def unpack_shared_vertices(
    shared: Ui64Array,
    domain_min: Vec3f,
    domain_max: Vec3f,
) -> Vec3fArray:
    res = np.empty((shared.size, 3), dtype=np.float32)
    res[:, 2] = (shared >> 42 & 0x3FFFFF) / 0x3FFFFF
    res[:, 1] = (shared >> 21 & 0x1FFFFF) / 0x1FFFFF
    res[:, 0] = (shared & 0x1FFFFF) / 0x1FFFFF

    res *= domain_max - domain_min
    res += domain_min
    return res


@dataclass
class CodecParams:
    min_xyz: Vec3f
    scale_xyz: Vec3f


@dataclass(init=False)
class HavokSection:
    domain_min: Vec3f
    domain_max: Vec3f
    codec_params: CodecParams
    firstPackedVertexIndex: int
    firstSharedVertexIndex: int
    firstPrimitiveIndex: int
    numPackedVertices: int
    numPrimitives: int
    firstDataRunIndex: int
    numDataRuns: int

    def __init__(self, section: dict[bytes, TagObject]):
        self.domain_min = np.array(
            [x.value for x in section[b'domain'].value[b'min'].value[:3]],
            dtype=np.float32,
        )
        self.domain_max = np.array(
            [x.value for x in section[b'domain'].value[b'max'].value[:3]],
            dtype=np.float32,
        )
        codec_values = [item.value for item in section[b'codecParms'].value]
        self.codec_params = CodecParams(
            min_xyz=np.array(codec_values[:3], dtype=np.float32),
            scale_xyz=np.array(codec_values[3:6], dtype=np.float32),
        )
        self.firstPackedVertexIndex = section[b'firstPackedVertexIndex'].value
        self.firstSharedVertexIndex = section[b'firstSharedVertexIndex'].value
        self.firstPrimitiveIndex = section[b'firstPrimitiveIndex'].value
        self.numPackedVertices = section[b'numPackedVertices'].value
        self.numPrimitives = section[b'numPrimitives'].value
        self.firstDataRunIndex = section[b'firstDataRunIndex'].value
        self.numDataRuns = section[b'numDataRuns'].value


@dataclass(init=False)
class HavokGeometry:
    name: str
    domain_min: Vec3f
    domain_max: Vec3f
    primitives: Quad4iArray
    sharedVertices: Ui64Array
    sharedVerticesIndex: Ui32Array
    packedVertices: Ui32Array
    sections: list[HavokSection]
    primitiveDataRuns: np.ndarray

    def __init__(self, name: str, meshTree: dict[bytes, TagObject]):
        self.name = name
        self.domain_min = np.array(
            [x.value for x in meshTree[b'domain'].value[b'min'].value[:3]],
            dtype=np.float32,
        )
        self.domain_max = np.array(
            [x.value for x in meshTree[b'domain'].value[b'max'].value[:3]],
            dtype=np.float32,
        )
        self.primitives = np.stack(
            [[_i.value for _i in x.value[b'indices'].value] for x in meshTree[b'primitives'].value],
        )
        self.sharedVerticesIndex = np.array([x.value for x in meshTree[b'sharedVerticesIndex'].value], dtype=np.uint32)

        self.sections = []
        for sec in meshTree[b'sections'].value:
            self.sections.append(HavokSection(sec.value))

        self.sharedVertices = np.array([x.value for x in meshTree[b'sharedVertices'].value], dtype=np.uint64)
        self.packedVertices = np.array([x.value for x in meshTree[b'packedVertices'].value], dtype=np.uint32)

        self.verts: Vec3fArray = np.empty(
            (self.packedVertices.size + self.sharedVertices.size, 3),
            dtype=np.float32,
        )
        self.verts[self.packedVertices.size :] = unpack_shared_vertices(self.sharedVertices, self.domain_min, self.domain_max)
        for sec in self.sections:
            if sec.numPackedVertices > 0:
                packed_slice = self.packedVertices[sec.firstPackedVertexIndex : sec.firstPackedVertexIndex + sec.numPackedVertices]
                self.verts[sec.firstPackedVertexIndex : sec.firstPackedVertexIndex + sec.numPackedVertices] = unpack_packed_vertices(
                    packed_slice,
                    sec.codec_params,
                    sec.domain_min,
                    sec.domain_max,
                )

            prim_slice = self.primitives[sec.firstPrimitiveIndex : sec.firstPrimitiveIndex + sec.numPrimitives]

            mask = prim_slice >= sec.numPackedVertices
            shared_indices = prim_slice - sec.numPackedVertices
            prim_slice[mask] = self.packedVertices.size + self.sharedVerticesIndex[sec.firstSharedVertexIndex + shared_indices[mask]]
            prim_slice[~mask] += sec.firstPackedVertexIndex

    @property
    def tris(self):
        """
        fan triangulate quads into tris
        handle degenerated quads
        """
        _tris = self.primitives[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
        dupes = (_tris[:, 0] == _tris[:, 1]) | (_tris[:, 1] == _tris[:, 2]) | (_tris[:, 0] == _tris[:, 2])
        return _tris[~dupes]


def read_geoms_from_havok(f: IO[bytes]) -> list[HavokGeometry]:
    assert TagReader.checkIO(f) == TagFileType.Object
    root_tag = TagReader.fromIO(f)

    assert len(root_tag.value[b'namedVariants'].value) == 1
    vals: list[TagObject] = root_tag.value[b'namedVariants'].value[0].value[b'variant'].value.value[b'resourceHandles'].value

    out = []
    for val in vals:
        if val.value.value[b'name'].value != 'Collision Physics Data':
            continue
        subVal = val.value.value[b'variant'].value.value
        if b'bodyCinfos' in subVal:
            for subSubVal in subVal[b'bodyCinfos'].value:
                if b'data' in subSubVal.value[b'shape'].value.value:
                    name = subSubVal.value[b'name'].value
                    out.append(
                        HavokGeometry(
                            name,
                            subSubVal.value[b'shape'].value.value[b'data'].value.value[b'meshTree'].value,
                        )
                    )

    return out
