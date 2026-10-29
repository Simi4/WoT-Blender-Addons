from ..LoadDataMesh import LoadDataMesh, PrimitiveGroup

__all__ = ('LoadDataMesh_v2', 'PrimitiveGroup')


class LoadDataMesh_v2(LoadDataMesh):
    """Compatibility wrapper using the map loader's coordinate convention."""

    def __init__(self, file, vertices_name: str, primitive_name: str):
        super().__init__(
            file,
            vertices_name,
            primitive_name,
            uv2_name=None,
            deduplicate=False,
            transform_coordinates=False,
        )
