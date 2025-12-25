
# imports
import shutil
import threading
from pathlib import Path
from zipfile import ZipFile

# blender imports
import bpy  # type: ignore
from bpy.app import tempdir  # type: ignore

# local imports
from ..common import addon_prefs
from .compiled_space.universal_space import UniversalResMgr, UniversalSpace
from .loader_context import LoaderContext
from .objects_loader import load_objects
from .terrain_loader import load_terrain
from .utils import GlobalVars


def extract_space_pkg(res_pkg_path: Path, geometry: str) -> str:
    space_name = geometry.split('/')[1]
    pkg_abs_path = res_pkg_path / f'{space_name}.pkg'
    pkg_bin_abs_path = res_pkg_path / f'{space_name}_bin.pkg'

    wot_tmp_1 = Path(tempdir) / 'wot'

    with ZipFile(pkg_abs_path, 'r') as zfile:
        zfile.extractall(wot_tmp_1)

    if pkg_bin_abs_path.is_file():
        with ZipFile(pkg_bin_abs_path, 'r') as zfile:
            zfile.extractall(wot_tmp_1)

    return wot_tmp_1


def parse_package(pkg_filepath_dict: dict, pkg_path: Path, formats_to_load: tuple[str]):
    if not pkg_path.is_file():
        return
    zfile = ZipFile(pkg_path, 'r')
    for fname in zfile.namelist():
        fname_lower = fname.lower()
        if fname_lower.endswith(formats_to_load):
            pkg_filepath_dict[fname_lower] = (zfile, fname)


def get_packages(res_pkg_path: Path, geometry: str) -> dict[str, tuple[ZipFile, str]]:
    space_name = geometry.split('/')[1]

    formats_to_load = ['.dds']
    if addon_prefs().map_viewer_load_objects:
        formats_to_load.append('.primitives_processed')
        formats_to_load.append('.atlas_processed')
        formats_to_load.append('.visual')
        formats_to_load.append('.primitives')

    formats_to_load = tuple(formats_to_load)

    pkg_filepath_dict = {}
    parse_package(pkg_filepath_dict, res_pkg_path / f'{space_name}.pkg', formats_to_load)
    parse_package(pkg_filepath_dict, res_pkg_path / f'{space_name}_bin.pkg', formats_to_load)
    parse_package(pkg_filepath_dict, res_pkg_path / 'particles.pkg', formats_to_load)
    for pkg_path in res_pkg_path.glob('shared*.pkg'):
        if '_hd-' in pkg_path.name:
            continue
        parse_package(pkg_filepath_dict, pkg_path, formats_to_load)

    return pkg_filepath_dict


def load(scene, map_info: dict, res_pkg_path: Path):
    if 'Map Collection' not in bpy.data.collections:
        map_col = bpy.data.collections.new('Map Collection')
        scene.collection.children.link(map_col)
    else:
        map_col = bpy.data.collections['Map Collection']

    geometry = map_info['geometry']
    wot_tmp_1 = extract_space_pkg(res_pkg_path, geometry)
    pkg_filepath_dict = get_packages(res_pkg_path, geometry)

    res_mgr = UniversalResMgr(pkg_filepath_dict, wot_tmp_1)
    space = UniversalSpace.from_space_dir(wot_tmp_1 / geometry, GlobalVars.wot_version, GlobalVars.wot_realm, res_mgr)

    loader_ctx = LoaderContext(res_mgr, space, map_info)
    if addon_prefs().map_viewer_load_objects:
        models_col = bpy.data.collections.new('Static Models Collection')
        map_col.children.link(models_col)
        load_objects(models_col, loader_ctx)

    terrain_col = bpy.data.collections.new('Terrain Collection')
    map_col.children.link(terrain_col)
    load_terrain(terrain_col, loader_ctx)

    def cleanup():
        shutil.rmtree(wot_tmp_1, ignore_errors=True)

    # async cleanup
    threading.Thread(target=cleanup).start()
