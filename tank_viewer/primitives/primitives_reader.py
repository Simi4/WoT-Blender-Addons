__all__ = ("PrimitiveGroup", "PrimitivesReader")

import logging
from dataclasses import dataclass
from struct import unpack_from

import numpy as np

from .vertex_types import (
    NormalFormat,
    vt_SET3_XYZNUVIIIWWTBPC,
    vt_SET3_XYZNUVTBPC,
    vt_SET3_XYZNUVPC,
    vt_XYZNUV,
    vt_XYZNUVIIIWWTB,
    vt_XYZNUVTB,
)


logger = logging.getLogger("primitives")


@dataclass(frozen=True, slots=True)
class PrimitiveGroup:
    start_index: int
    primitive_count: int
    start_vertex: int
    vertex_count: int


@dataclass(frozen=True, slots=True)
class _PackedSection:
    position: int
    length: int


def unpack_normals_8_8_8(packed_array: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed_array, dtype=np.uint32)
    packed_z = (packed >> 16) & 0xFF
    packed_y = (packed >> 8) & 0xFF
    packed_x = packed & 0xFF
    channels = np.stack((packed_x, packed_y, packed_z), axis=-1).astype(np.int32)

    negative_mask = channels > 0x7F
    normals = np.empty_like(channels, dtype=np.float32)
    normals[negative_mask] = (
        -((channels[negative_mask] & 0x7F).astype(np.float32)) / 127.0
    )
    normals[~negative_mask] = (
        (channels[~negative_mask] ^ 0x7F).astype(np.float32)
    ) / 127.0
    return normals


def unpack_normals_11_11_10(packed_array: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed_array, dtype=np.uint32)
    packed_x = packed & 0x7FF
    packed_y = (packed >> 11) & 0x7FF
    packed_z = (packed >> 22) & 0x3FF
    channels = np.stack((packed_x, packed_y, packed_z), axis=-1).astype(np.int32)

    midpoints = np.array((0x3FF, 0x3FF, 0x1FF), dtype=np.int32)
    divisors = np.array((1023.0, 1023.0, 511.0), dtype=np.float32)
    negative_mask = channels > midpoints
    negative_values = -((channels & midpoints).astype(np.float32)) / divisors
    positive_values = ((channels ^ midpoints).astype(np.float32)) / divisors
    return np.where(negative_mask, negative_values, positive_values)


class PrimitivesReader:
    """Read geometry streams and primitive groups from a primitives file."""

    def __init__(
        self,
        data: bytes,
        vertices_name: str = "vertices",
        primitive_name: str = "indices",
        uv2_name: str | None = "",
    ):
        self._buffer = memoryview(data)
        self._packed_sections: dict[str, _PackedSection] = {}

        self.is_processed = False
        self.primitive_groups: list[PrimitiveGroup] = []
        self.vertices = None
        self.normals = None
        self.indices = None
        self.uv_list = None
        self.uv2_list = None

        header = unpack_from("<I", self._buffer)[0]
        assert header == 0x42A14E65

        self._load_packed_sections()

        if uv2_name is None:
            candidate = vertices_name[:-8] + "uv2"
            uv2_name = candidate if candidate in self._packed_sections else ""

        self._load_geometry(
            self._packed_sections[primitive_name].position,
            self._packed_sections[vertices_name].position,
        )

        if uv2_name and uv2_name in self._packed_sections:
            section = self._packed_sections[uv2_name]
            self._load_uv2(section.position, section.length)

    def get_vertices_by_id(self, group_index: int) -> np.ndarray:
        return self.vertices[self._get_vertex_slice(group_index)]

    def get_normals_by_id(self, group_index: int) -> np.ndarray:
        return self.normals[self._get_vertex_slice(group_index)]

    def get_indices_by_id(self, group_index: int) -> np.ndarray:
        group = self.primitive_groups[group_index]
        start = group.start_index // 3
        indices = self.indices[start : start + group.primitive_count]
        return indices - group.start_vertex

    def get_uv_by_id(self, group_index: int) -> np.ndarray:
        return self.uv_list[self._get_vertex_slice(group_index)]

    def get_uv2_by_id(self, group_index: int) -> np.ndarray:
        return self.uv2_list[self._get_vertex_slice(group_index)]

    def _get_vertex_slice(self, group_index: int) -> slice:
        group = self.primitive_groups[group_index]
        return slice(group.start_vertex, group.start_vertex + group.vertex_count)

    def _read_fixed_string(self, offset: int, length: int = 64) -> bytes:
        return self._buffer[offset : offset + length].tobytes().split(b"\x00", 1)[0]

    def _load_packed_sections(self) -> None:
        table_start = unpack_from("<l", self._buffer, len(self._buffer) - 4)[0]
        offset = len(self._buffer) - 4 - table_start
        position = 4

        while offset + 4 <= len(self._buffer):
            section_size = unpack_from("<I", self._buffer, offset)[0]
            offset += 20
            if offset + 4 > len(self._buffer):
                break

            section_name_length = unpack_from("<I", self._buffer, offset)[0]
            offset += 4
            section_name = (
                self._buffer[offset : offset + section_name_length]
                .tobytes()
                .decode("utf-8")
            )
            offset += section_name_length

            if any(
                item in section_name
                for item in ("vertices", "indices", "uv2", "colour")
            ):
                self._packed_sections[section_name] = _PackedSection(
                    position=position,
                    length=section_size,
                )

            position += section_size + (-section_size % 4)
            offset += -section_name_length % 4

    @staticmethod
    def _get_vertex_type(
        vertices_format: bytes, processed_format: bytes
    ) -> type | None:
        match processed_format:
            case vt_SET3_XYZNUVIIIWWTBPC.V_TYPE:
                return vt_SET3_XYZNUVIIIWWTBPC
            case vt_SET3_XYZNUVTBPC.V_TYPE:
                return vt_SET3_XYZNUVTBPC
            case vt_SET3_XYZNUVPC.V_TYPE:
                return vt_SET3_XYZNUVPC

        if vt_XYZNUVIIIWWTB.V_TYPE in vertices_format:
            return vt_XYZNUVIIIWWTB
        if vt_XYZNUVTB.V_TYPE in vertices_format:
            return vt_XYZNUVTB
        if vt_XYZNUV.V_TYPE in vertices_format:
            return vt_XYZNUV
        return None

    def _load_geometry(self, indices_position: int, vertices_position: int) -> None:
        index_format = self._read_fixed_string(indices_position).decode("utf-8")
        index_count = unpack_from("<I", self._buffer, indices_position + 64)[0]
        group_count = unpack_from("<H", self._buffer, indices_position + 68)[0]

        if index_format == "list":
            index_dtype = "<u2"
        elif index_format == "list32":
            index_dtype = "<u4"
        else:
            raise ValueError(f"Unsupported index format: {index_format}")

        group_position = (
            indices_position + index_count * np.dtype(index_dtype).itemsize + 72
        )
        for _ in range(group_count):
            values = unpack_from("<4I", self._buffer, group_position)
            self.primitive_groups.append(PrimitiveGroup(*values))
            group_position += 16

        vertices_format = self._read_fixed_string(vertices_position)
        processed_format = b""
        offset = vertices_position + 64
        if b"BPVT" in vertices_format:
            self.is_processed = True
            offset += 4
            processed_format = self._read_fixed_string(offset)
            offset += 64

        vertex_count = unpack_from("<l", self._buffer, offset)[0]
        offset += 4
        vertex_type = self._get_vertex_type(vertices_format, processed_format)
        if vertex_type is None:
            logger.warning(
                "processed_format=%s; vertices_format=%s",
                processed_format,
                vertices_format,
            )
            return

        vertex_data = np.frombuffer(
            self._buffer,
            dtype=vertex_type.DTYPE,
            count=vertex_count,
            offset=offset,
        )

        self.vertices = vertex_data["pos"]
        self.uv_list = vertex_data["uv"].copy()
        self.uv_list[:, 1] = 1.0 - self.uv_list[:, 1]

        if vertex_type.NORMAL_FMT == NormalFormat.PACKED_8_8_8:
            self.normals = unpack_normals_8_8_8(vertex_data["packed_normal"])
        elif vertex_type.NORMAL_FMT == NormalFormat.PACKED_11_11_10:
            self.normals = unpack_normals_11_11_10(vertex_data["packed_normal"])
        else:
            self.normals = vertex_data["normal"]

        if vertex_type.IS_SKINNED:
            self.vertices = self.vertices.copy()
            self.normals = self.normals.copy()
            self.vertices[:, 2] *= -1.0
            self.normals[:, 2] *= -1.0

        indices = np.frombuffer(
            self._buffer,
            dtype=np.dtype((index_dtype, 3)),
            count=index_count // 3,
            offset=indices_position + 72,
        )
        self.indices = indices[:, [2, 1, 0]].copy()

    def _load_uv2(self, uv2_position: int, uv2_length: int) -> None:
        if not self.is_processed:
            self.uv2_list = np.frombuffer(
                self._buffer,
                dtype=("<f4", 2),
                count=uv2_length // 8,
                offset=uv2_position,
            ).copy()
        else:
            uv2_subname = self._read_fixed_string(uv2_position)
            uv2_format = b""
            offset = uv2_position + 64
            if b"BPVS" in uv2_subname:
                offset += 4
                uv2_format = self._read_fixed_string(offset)
                offset += 64

            uv2_count = unpack_from("<I", self._buffer, offset)[0]
            offset += 4
            if uv2_format != b"set3/uv2pc":
                logger.warning("uv2_format=%s", uv2_format)
                return

            self.uv2_list = np.frombuffer(
                self._buffer,
                dtype=("<f4", 2),
                count=uv2_count,
                offset=offset,
            ).copy()

        self.uv2_list[:, 1] = 1.0 - self.uv2_list[:, 1]
