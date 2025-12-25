
# imports
from dataclasses import dataclass
from os import SEEK_CUR
from struct import unpack
from typing import IO

# local imports
from .loader_context import LoaderContext


@dataclass
class AtlasCoords:
    x0: int
    x1: int
    y0: int
    y1: int
    path: str


def parse_atlas(f: IO[bytes]) -> list[AtlasCoords] | None:
    atlasParts = []

    version, atlas_width, atlas_height, unused1, magic, unused2, dds_chunk_size = unpack('<6IQ', f.read(32))

    if magic == 0:
        # TODO: need to investigate this case
        return None

    assert version == 1
    assert unused1 in (0, 1)
    assert magic == int.from_bytes(b'BCVT', 'little')
    assert unused2 == 1

    f.seek(dds_chunk_size, SEEK_CUR)

    while True:
        data = f.read(16)
        if data is None or len(data) != 16:
            break
        x0, x1, y0, y1 = unpack('<4I', data)
        path = b''
        tmpChar = f.read(1)
        while tmpChar != b'\x00':
            path += tmpChar
            tmpChar = f.read(1)

        path = path.decode('utf-8')
        path = path[:-4] + '.dds'
        atlasParts.append(AtlasCoords(x0, x1, y0, y1, path))

    return atlasParts


def load_atlas(ctx: LoaderContext, name: str) -> list[AtlasCoords] | None:
    if not ctx.res_mgr.exists(name):
        return None

    with ctx.res_mgr.open(name) as f:
        return parse_atlas(f)


def load_atlas_dds(name: str, indexes: list[float]) -> dict[int, AtlasCoords]:
    atlasParts = {}

    for i in indexes:
        atlasParts[int(i)] = AtlasCoords(0, 0, 0, 0, name)

    return atlasParts
