import gettext
import sqlite3
from xml.etree import ElementTree as ET
from dataclasses import dataclass

from .common import *
from .ResourceManager import ResourceManager


__all__ = ('WotDataManager', 'g_LevelsL10n', 'g_NationsL10n', 'g_CategoryL10n')


#####################################################################
# TankListPaths

class TankListPaths:
    JAPAN = Path('scripts/item_defs/vehicles/japan')
    CHINA = Path('scripts/item_defs/vehicles/china')
    CZECH = Path('scripts/item_defs/vehicles/czech')
    FRANCE = Path('scripts/item_defs/vehicles/france')
    GERMANY = Path('scripts/item_defs/vehicles/germany')
    UK = Path('scripts/item_defs/vehicles/uk')
    USA = Path('scripts/item_defs/vehicles/usa')
    USSR = Path('scripts/item_defs/vehicles/ussr')
    SWEDEN = Path('scripts/item_defs/vehicles/sweden')
    POLAND = Path('scripts/item_defs/vehicles/poland')
    ITALY = Path('scripts/item_defs/vehicles/italy')
    INTUNION = Path('scripts/item_defs/vehicles/intunion')
    INDEX = {
        1: JAPAN, 2: CHINA,
        3: CZECH, 4: FRANCE,
        5: GERMANY, 6: UK,
        7: USA, 8: USSR,
        9: SWEDEN, 10: POLAND,
        11: ITALY, 12: INTUNION
    }


#####################################################################
# MoFiles

class MoFiles:
    JAPAN_MO = Path('lc_messages/japan_vehicles.mo')
    JAPAN_ID = '#japan_vehicles'
    CHINA_MO = Path('lc_messages/china_vehicles.mo')
    CHINA_ID = '#china_vehicles'
    CZECH_MO = Path('lc_messages/czech_vehicles.mo')
    CZECH_ID = '#czech_vehicles'
    FRANCE_MO = Path('lc_messages/france_vehicles.mo')
    FRANCE_ID = '#france_vehicles'
    GERMANY_MO = Path('lc_messages/germany_vehicles.mo')
    GERMANY_ID = '#germany_vehicles'
    UK_MO = Path('lc_messages/gb_vehicles.mo')
    UK_ID = '#gb_vehicles'
    USA_MO = Path('lc_messages/usa_vehicles.mo')
    USA_ID = '#usa_vehicles'
    USSR_MO = Path('lc_messages/ussr_vehicles.mo')
    USSR_ID = '#ussr_vehicles'
    SWEDEN_MO = Path('lc_messages/sweden_vehicles.mo')
    SWEDEN_ID = '#sweden_vehicles'
    POLAND_MO = Path('lc_messages/poland_vehicles.mo')
    POLAND_ID = '#poland_vehicles'
    ITALY_MO = Path('lc_messages/italy_vehicles.mo')
    ITALY_ID = '#italy_vehicles'
    INTUNION_MO = Path('lc_messages/intunion_vehicles.mo')
    INTUNION_ID = '#intunion_vehicles'
    MENU_MO = Path('lc_messages/menu.mo')
    MENU_ID = '#menu'
    CUSTOMIZATION_MO = Path('lc_messages/vehicle_customization.mo')
    CUSTOMIZATION_ID = '#vehicle_customization'
    ALL = {
        JAPAN_ID: [JAPAN_MO, None],
        CHINA_ID: [CHINA_MO, None],
        CZECH_ID: [CZECH_MO, None],
        FRANCE_ID: [FRANCE_MO, None],
        GERMANY_ID: [GERMANY_MO, None],
        UK_ID: [UK_MO, None],
        USA_ID: [USA_MO, None],
        USSR_ID: [USSR_MO, None],
        SWEDEN_ID: [SWEDEN_MO, None],
        POLAND_ID: [POLAND_MO, None],
        ITALY_ID: [ITALY_MO, None],
        INTUNION_ID: [INTUNION_MO, None],
        MENU_ID: [MENU_MO, None],
        CUSTOMIZATION_ID: [CUSTOMIZATION_MO, None]
    }

    def load(self, wot_res_path: str):
        for key, value in self.ALL.items():
            mo_path = Path(wot_res_path) / 'text' / value[0]
            if not mo_path.is_file():
                mo_path = Path(wot_res_path) / 'text' / 'ru' / value[0]
                if not mo_path.is_file():
                    continue
            self.ALL[key][1] = gettext.GNUTranslations(mo_path.open('rb'))

    def _(self, str_):
        try:
            return self.ALL[str_.split(':')[0]][1].gettext(str_.split(':')[1])
        except Exception:
            return str_.split(':')[1]


#####################################################################
# g_MoFiles

g_MoFiles = MoFiles()


#####################################################################
# LevelListL10n

class LevelsL10n:
    _ALL = 'all'
    _FIRST = '1'
    _SECOND = '2'
    _THIRD = '3'
    _FOURTH = '4'
    _FIFTH = '5'
    _SIXTH = '6'
    _SEVENTH = '7'
    _EIGHTH = '8'
    _NINTH = '9'
    _TENTH = '10'
    _ELEVENTH = '11'
    ALL = {
        _ALL: 0, _FIRST: 1,
        _SECOND: 2, _THIRD: 3,
        _FOURTH: 4, _FIFTH: 5,
        _SIXTH: 6, _SEVENTH: 7,
        _EIGHTH: 8, _NINTH: 9,
        _TENTH: 10, _ELEVENTH: 11
    }
    INDEX = {}

    def load(self):
        for key, value in self.ALL.items():
            self.INDEX[value] = g_MoFiles._(f'#menu:levels/{key}')


#####################################################################
# g_LevelsL10n

g_LevelsL10n = LevelsL10n()


#####################################################################
# NationsL10n

class NationsL10n:
    _ALL = 'all'
    _JAPAN = 'japan'
    _CHINA = 'china'
    _CZEH = 'czech'
    _FRANCE = 'france'
    _GERMANY = 'germany'
    _UK = 'uk'
    _USA = 'usa'
    _USSR = 'ussr'
    _SWEDEN = 'sweden'
    _POLAND = 'poland'
    _ITALY = 'italy'
    _INTUNION = 'intunion'
    ALL = {
        _ALL: 0, _JAPAN: 1,
        _CHINA: 2, _CZEH: 3,
        _FRANCE: 4, _GERMANY: 5,
        _UK: 6, _USA: 7,
        _USSR: 8, _SWEDEN: 9,
        _POLAND: 10, _ITALY: 11,
        _INTUNION: 12
    }
    INDEX = {}

    def load(self):
        for key, value in self.ALL.items():
            self.INDEX[value] = g_MoFiles._(f'#menu:nations/{key}')


#####################################################################
# g_NationsL10ns

g_NationsL10n = NationsL10n()


#####################################################################
# CategoryL10n

class CategoryL10n:
    _ALL = 'all'
    _LIGHT_TANK = 'lightTank'
    _MEDIUM_TANK = 'mediumTank'
    _HEAVY_TANK = 'heavyTank'
    _AT_SPG = 'AT-SPG'
    _SPG = 'SPG'
    ALL = {
        _ALL: 0, _LIGHT_TANK: 1,
        _MEDIUM_TANK: 2, _HEAVY_TANK: 3,
        _AT_SPG: 4, _SPG: 5
    }
    INDEX = {}

    def load(self):
        for key, value in self.ALL.items():
            self.INDEX[value] = g_MoFiles._(f'#menu:carousel_tank_filter/{key}')


#####################################################################
# g_CategoryL10n

g_CategoryL10n = CategoryL10n()


#####################################################################
# helper classes

@dataclass(init=False)
class ModelArmor:
    armors: dict[str, float]

    def __init__(self, elem: ET.Element):
        self.armors = {}
        for it in elem.iterfind('*'):
            self.armors[it.tag] = float(it.text)


@dataclass(init=False)
class ModelSet:
    undamaged_model: str
    destroyed_model: str

    def __init__(self, elem):
        self.undamaged_model = StrToFilePath(elem.findtext('undamaged'))
        self.destroyed_model = StrToFilePath(elem.findtext('destroyed'))


@dataclass(init=False)
class SegmentModels:
    segmentModelLeft: str
    segmentModelRight: str
    segment2ModelLeft: str
    segment2ModelRight: str

    def __init__(self, elem):
        self.segmentModelLeft = StrToFilePath(elem.findtext('segmentModelLeft'))
        self.segmentModelRight = StrToFilePath(elem.findtext('segmentModelRight'))
        self.segment2ModelLeft = StrToFilePath(elem.findtext('segment2ModelLeft'))
        self.segment2ModelRight = StrToFilePath(elem.findtext('segment2ModelRight'))

    def has_two_segments(self) -> bool:
        return bool(self.segment2ModelLeft and self.segment2ModelRight)


@dataclass(init=False)
class TrackPair:
    left: str
    right: str
    right: str
    segmentLength: float
    segmentOffset: float
    segment2Offset: float
    sets: dict[str, SegmentModels]

    def __init__(self, elem):
        self.sets = { 'default' : SegmentModels(elem) }

        self.left = StrToFilePath(elem.findtext('left'))
        self.right = StrToFilePath(elem.findtext('right'))

        self.segmentLength = float(elem.findtext('segmentLength', 0))
        self.segmentOffset = float(elem.findtext('segmentOffset', 0))

        # TODO: why don't we use it?
        self.segment2Offset = float(elem.findtext('segment2Offset', 0))

        for elem_set in elem.findall('modelSets/*'):
            self.sets[elem_set.tag] = SegmentModels(elem_set)


@dataclass(init=False)
class ChassisSpline:
    trackPairs: list[TrackPair]

    def __init__(self, elem: ET.Element):
        self.trackPairs = []

        if elem.find('trackPair') is None:
            self.trackPairs.append(TrackPair(elem))

        for trackPair_elem in elem.iterfind('trackPair'):
            self.trackPairs.append(TrackPair(trackPair_elem))


@dataclass(init=False)
class ChassisDesc:
    collision_model: str
    hullPosition: Vector
    camouflage_tiling: Vector
    splineDesc: ChassisSpline
    sets: dict[str, ModelSet]
    armors: ModelArmor

    def __init__(self, elem):
        self.collision_model = StrToFilePath(elem.findtext('hitTester/collisionModelClient') or \
                                             elem.findtext('hitTester/collisionModel'))
        self.hullPosition = StrToVector(elem.findtext('hullPosition')).xzy
        self.splineDesc = elem.find('splineDesc') and ChassisSpline(elem.find('splineDesc'))

        self.armors = ModelArmor(elem.find('armor')) if elem.find('armor') is not None else None

        self.sets = { 'default' : ModelSet(elem.find('models')) }

        for elem_set in elem.findall('models/sets/*'):
            self.sets[elem_set.tag] = ModelSet(elem_set)


@dataclass(init=False)
class HullDesc:
    chassis: ChassisDesc
    collision_model: str
    turretPositions: Vector
    camouflage_tiling: Vector
    sets: dict[str, ModelSet]
    armors: ModelArmor

    def __init__(self, elem, chassis: ChassisDesc):
        self.chassis = chassis
        self.collision_model = StrToFilePath(elem.findtext('hitTester/collisionModelClient') or \
                                             elem.findtext('hitTester/collisionModel'))
        self.turretPositions = StrToVector(elem.findtext('turretPositions/turret')).xzy + chassis.hullPosition
        # TODO: self.camouflage_tiling = StrToVector(elem.findtext('camouflage/tiling'))

        self.armors = ModelArmor(elem.find('armor'))

        self.sets = { 'default' : ModelSet(elem.find('models')) }

        for elem_set in elem.findall('models/sets/*'):
            self.sets[elem_set.tag] = ModelSet(elem_set)


@dataclass(init=False)
class GunDesc:
    userString: str
    l10n_name: str
    collision_model: str
    camouflage_tiling: Vector
    sets: dict[str, ModelSet]
    armors: ModelArmor

    def __init__(self, elem, turret):
        self.userString = turret.userString.split(':')[0] + ':' + elem.tag
        self.l10n_name = g_MoFiles._(self.userString)

        # TODO: self.camouflage_tiling = StrToVector(elem.findtext('camouflage/tiling'))
        self.collision_model = StrToFilePath(elem.findtext('hitTester/collisionModelClient') or \
                                             elem.findtext('hitTester/collisionModel'))

        self.armors = ModelArmor(elem.find('armor'))

        self.sets = { 'default' : ModelSet(elem.find('models')) }

        for elem_set in elem.findall('models/sets/*'):
            self.sets[elem_set.tag] = ModelSet(elem_set)


@dataclass(init=False)
class TurretDesc:
    userString: str
    l10n_name: str
    collision_model: str
    gunPosition: Vector
    camouflage_tiling: Vector
    sets: dict[str, ModelSet]
    guns: dict[str, GunDesc]
    armors: ModelArmor

    def __init__(self, elem):
        # TODO: self.camouflage_tiling = StrToVector(elem.findtext('camouflage/tiling'))
        self.gunPosition = StrToVector(elem.findtext('gunPosition')).xzy
        self.collision_model = StrToFilePath(elem.findtext('hitTester/collisionModelClient') or \
                                             elem.findtext('hitTester/collisionModel'))
        self.userString = elem.findtext('userString').strip()
        self.l10n_name = g_MoFiles._(self.userString)

        self.guns = {}
        for gun_elem in elem.find('guns'):
            gdesc = GunDesc(gun_elem, self)
            self.guns[gdesc.l10n_name] = gdesc

        self.armors = ModelArmor(elem.find('armor')) if elem.find('armor') is not None else None

        self.sets = { 'default' : ModelSet(elem.find('models')) }

        for elem_set in elem.findall('models/sets/*'):
            self.sets[elem_set.tag] = ModelSet(elem_set)


@dataclass(init=False)
class FullTankDesc:
    camouflage_tiling: Vector
    camouflage_exclusionMask: Vector
    chassises: dict[str, ChassisDesc]
    hulls: dict[str, HullDesc]
    turrets0: dict[str, TurretDesc]

    def __init__(self, elem):
        # TODO: self.camouflage_tiling = StrToVector(elem.findtext('camouflage/tiling'))
        # TODO: self.camouflage_exclusionMask = StrToVector(elem.findtext('camouflage/exclusionMask'))

        self.chassises = {}
        for chassis_elem in elem.find('chassis'):
            c_desc = ChassisDesc(chassis_elem)
            self.chassises[chassis_elem.tag] = c_desc

        self.hulls = {}
        self.hulls['default'] = HullDesc(elem.find('hull'), list(self.chassises.values())[0])
        if elem.find('hull/variants'):
            if chassis_name := elem.findtext('hull/variants/hull2/chassis'):
                chassis_name = chassis_name.strip()
                self.hulls['hull2'] = HullDesc(elem.find('hull/variants/hull2'), self.chassises[chassis_name])
            if chassis_name := elem.findtext('hull/variants/hull3/chassis'):
                chassis_name = chassis_name.strip()
                self.hulls['hull3'] = HullDesc(elem.find('hull/variants/hull3'), self.chassises[chassis_name])

        self.turrets0 = {}
        for turret_elem in elem.find('turrets0'):
            tdesc = TurretDesc(turret_elem)

            # Fix for some SPG-guns:
            tmp = tdesc.l10n_name
            i = 0
            while tdesc.l10n_name in self.turrets0:
                i += 1
                tdesc.l10n_name = f'{tmp} ({i})'

            self.turrets0[tdesc.l10n_name] = tdesc


@dataclass(init=False)
class WotDataManager:
    ''' Main class for working with WoT client '''

    wot_res_path: Path
    wot_version: tuple
    res_mgr: ResourceManager
    conn: sqlite3.Connection
    sqlite3_cursor: sqlite3.Cursor

    @classmethod
    def add_client(cls, wot_dir_path: str):
        wot_res_path = Path(wot_dir_path) / 'res'
        wot_version_path = Path(wot_dir_path) / 'version.xml'

        if not wot_res_path.is_dir():
            return None

        if not wot_version_path.is_file():
            return None

        wot_version = cls.get_version(wot_version_path)
        return cls(wot_res_path, wot_version)

    def __init__(self, wot_res_path: Path, wot_version: tuple):
        self.wot_res_path = wot_res_path
        self.wot_version = wot_version

        self.conn = sqlite3.connect(':memory:')
        self.sqlite3_cursor = self.conn.cursor()
        self.sqlite3_cursor.execute('CREATE TABLE Vehicles (id INTEGER PRIMARY KEY NOT NULL, tank_nation INTEGER, tank_type_id INTEGER, tank_props INTEGER, tank_name TEXT, tank_name_in_res TEXT, tank_level INTEGER) STRICT;')
        self.sqlite3_cursor.execute('CREATE TABLE Styles (id INTEGER PRIMARY KEY NOT NULL, vehicle_id INTEGER, style_name TEXT, models_set TEXT, FOREIGN KEY (vehicle_id) REFERENCES Vehicles(id)) STRICT;')

        self.res_mgr = ResourceManager(self.wot_res_path)

        self.loadTranslations()
        self.loadVehicleDictionary()
        self.loadCustomization()

    def __del__(self):
        logger.debug('__del__')
        self.sqlite3_cursor.close()
        self.conn.close()
        del self.res_mgr

    @staticmethod
    def get_version(filename: Path) -> tuple:
        ''' Get WoT version from version.xml '''
        tree = ET.parse(filename)
        elem = tree.getroot()
        # Examples:
        # <version> v.0.9.13 #68</version>
        # <version> v.0.9.15.0.1 #44</version>
        # <version> v.0.9.15.1 Common Test #172</version>
        _version = elem.findtext('version').strip()

        logger.info(f'WoT client: {_version}')

        if 'Supertest v.ST ' in _version:
            _version = _version.replace('Supertest v.ST ', 'v.')
        elif ' Common Test' in _version:
            _version = _version.replace(' Common Test', '')
        return tuple(map(int, _version.split('v.')[1].split('#')[0].strip().split('.')))


    def loadTranslations(self):
        g_MoFiles.load(self.wot_res_path)
        g_LevelsL10n.load()
        g_NationsL10n.load()
        g_CategoryL10n.load()


    def loadCustomization(self):
        listNodes = self.res_mgr.open_scripts_xml(Path('scripts/item_defs/customization/styles/list.xml'))
        if listNodes is None:
            return
        for node in listNodes.findall('itemGroup'):
            if modelsSet := node.findtext('style/modelsSet'):
                modelsSet = modelsSet.strip()
                if not modelsSet:
                    continue

                vehicles = node.findtext('vehicleFilter/include/vehicles')
                if not vehicles:
                    # NOTE: fix for Tanki 1.39 (Un02_Merkava_LP_3Dst_01)
                    vehicles = node.findtext('style/vehicleFilter/include/vehicles')

                if not vehicles:
                    logger.warning(f"empty filter for {modelsSet=}")
                    continue

                vehicles = vehicles.strip()
                for v in vehicles.split():
                    _, vehicle = v.split(':')

                    style_name = g_MoFiles._(node.findtext('style/userString').strip())
                    self.sqlite3_cursor.execute('INSERT INTO Styles(vehicle_id, style_name, models_set) VALUES ((SELECT id from Vehicles WHERE tank_name_in_res=?), ?, ?)',
                        (vehicle, style_name, modelsSet))

    def loadVehicleDictionary(self):
        for (idx, nation_path) in TankListPaths.INDEX.items():
            try:
                listNodes = self.res_mgr.open_scripts_xml(nation_path / 'list.xml')
            except Exception:
                continue

            if listNodes is None:
                continue

            for node in listNodes:
                try:
                    vehicle_userString = node.findtext('userString').strip()
                except Exception:
                    continue

                vehicle_props = 0

                vehicle_tags = node.findtext('tags').strip().split()
                if 'secret' in vehicle_tags:
                    vehicle_props |= 1

                    if vehicle_userString.startswith(('#igr_vehicles:', '#maps_training:')):
                        continue

                    if vehicle_userString.endswith(('_bot', '_fallout')):
                        continue

                    if node.tag.endswith(('_training', '_fallout')):
                        continue

                vehicle_category = vehicle_tags[0]
                if vehicle_category not in g_CategoryL10n.ALL:
                    vehicle_category = vehicle_tags[1]
                    if vehicle_category not in g_CategoryL10n.ALL:
                        continue

                vehicle_category = g_CategoryL10n.ALL[vehicle_category]
                vehicle_level = AsInt(node.findtext('level'))
                vehicle_l10n_name = g_MoFiles._(vehicle_userString)

                self.sqlite3_cursor.execute('INSERT INTO Vehicles(tank_nation, tank_type_id, tank_props, tank_name, tank_name_in_res, tank_level) VALUES (?, ?, ?, ?, ?, ?)', (
                    idx, vehicle_category, vehicle_props,
                    vehicle_l10n_name, node.tag, vehicle_level))


    def loadInfo(self, vehicle_id: int):
        nation_idx, xml_file = self.sqlite3_cursor.execute('SELECT tank_nation, tank_name_in_res FROM Vehicles WHERE id=?',
            (vehicle_id,)).fetchone()

        xml_path = TankListPaths.INDEX[nation_idx] / f'{xml_file}.xml'

        element = self.res_mgr.open_scripts_xml(xml_path)

        return FullTankDesc(element)
