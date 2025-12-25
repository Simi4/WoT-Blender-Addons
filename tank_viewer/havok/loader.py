
# imports
import logging
import os

# blender imports
import bpy

# local imports
from ..ResourceManager import ResourceManager
from .model_extractor import read_geoms_from_havok

logger = logging.getLogger(__name__)


def load_havok_file(res_mgr: ResourceManager, col: bpy.types.Collection, model: dict):
    filepath = model['File'].replace('.model', '.havok')

    logger.info(f'Start loading havok: {filepath}')
    havok_f = res_mgr.open_file(filepath)

    if not havok_f:
        logger.error('File not found!')
        return

    geoms = read_geoms_from_havok(havok_f)

    for i, geom in enumerate(geoms):
        ob_name = os.path.basename(filepath).replace('.', '_') + f'_{geom.name}_{i}'

        verts = geom.verts
        faces = geom.tris
        verts = verts[:, [0, 2, 1]]
        bmesh = bpy.data.meshes.new(f'Mesh_{ob_name}')
        bmesh.from_pydata(verts, [], faces)

        bmesh.validate()
        bmesh.update()

        ob = bpy.data.objects.new(ob_name, bmesh)

        try:
            ob['armor'] = model['Armors'].armors[geom.name[2:]]
        except Exception:
            pass

        ob.location = model['Position']
        col.objects.link(ob)
