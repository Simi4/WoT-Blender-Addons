
# imports
import gettext
import logging
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

# local imports
from ..common import *


logging.basicConfig()
logger = logging.getLogger(__name__)


class GlobalVars:
    wot_res_path: Path | None = None
    wot_version: str | None = None
    wot_realm: str | None = None
    arenas_mo_gettext: gettext.GNUTranslations | None = None

    @classmethod
    def clear(cls):
        cls.wot_res_path = None
        cls.wot_version = None
        cls.arenas_mo_gettext = None


def _a(str_):
    try:
        return GlobalVars.arenas_mo_gettext.gettext(f'{str_}/name')
    except Exception:
        return str_


def init(wot_res_path_: Path, wot_version_path_: Path):
    arenas_mo_path = wot_res_path_ / 'text' / 'lc_messages' / 'arenas.mo'
    if not arenas_mo_path.is_file():
        arenas_mo_path = wot_res_path_ / 'text' / 'ru' / 'lc_messages' / 'arenas.mo'

    if arenas_mo_path.is_file():
        arenas_mo_gettext = gettext.GNUTranslations(arenas_mo_path.open('rb'))
    else:
        arenas_mo_gettext = None

    version_xml_tree = ET.parse(wot_version_path_)
    version_xml_root = version_xml_tree.getroot()

    GlobalVars.wot_version = version_xml_root.findtext('version').strip()
    GlobalVars.wot_realm = version_xml_root.findtext('meta/realm', 'RU').strip()
    GlobalVars.wot_res_path = wot_res_path_
    GlobalVars.arenas_mo_gettext = arenas_mo_gettext

    logger.info(f'WoT client: {GlobalVars.wot_version} {GlobalVars.wot_realm} {GlobalVars.wot_res_path}')


def fini():
    GlobalVars.clear()


def open_xml(path):
    list_file_path = GlobalVars.wot_res_path / path
    if list_file_path.is_file():
        with list_file_path.open('rb') as f:
            return g_XmlUnpacker.read(f)
    else:
        scripts_pkg_path = GlobalVars.wot_res_path / 'packages' / 'scripts.pkg'
        assert scripts_pkg_path.is_file()
        with ZipFile(scripts_pkg_path, 'r') as scripts_pkg:
            with scripts_pkg.open(path, 'r') as f:
                return g_XmlUnpacker.read(f)


def load_maps_dictionary():
    """Maps List loader"""

    listNodes = open_xml('scripts/arena_defs/_list_.xml')

    maps_list = []
    for node in listNodes.findall('map'):
        if node.find('id') is None or node.find('name') is None:
            continue

        # map_id = AsInt(node.findtext('id'))
        map_name = node.findtext('name').strip()
        map_l10n_name = _a(map_name)
        maps_list.append((len(maps_list), map_name, map_l10n_name))

    # dirty hack to add hangars to map list
    hangars = ['hangar_v3', 'h02_mt_bday_2023', 'h33_battle_royale_2021', 'h33_comp7']
    for h_name in hangars:
        if (GlobalVars.wot_res_path / 'packages' / f'{h_name}.pkg').is_file():
            maps_list.append((len(maps_list), h_name, h_name))

    return maps_list


def load_map_info(map_name: str):
    """Map Info loader"""

    nodes = open_xml(f'scripts/arena_defs/{map_name}.xml')
    map_info = {}

    if nodes.find('geometry') is not None:
        map_info['geometry'] = nodes.findtext('geometry').strip()
    else:
        map_info['geometry'] = f'spaces/{map_name}'

    map_info['boundingBox'] = {
        'bottomLeft': StrToVector(nodes.findtext('boundingBox/bottomLeft')),
        'upperRight': StrToVector(nodes.findtext('boundingBox/upperRight')),
    }

    return map_info
