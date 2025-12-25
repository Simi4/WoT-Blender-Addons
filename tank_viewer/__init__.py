bl_info = {
    'name': 'World of Tanks - Map & Vehicle Viewer',
    'author': 'Simi4',
    'version': (0, 3, 0),
    'blender': (4, 3, 0),
    'location': 'View3D > Sidebar > TankViewer',
    'description': 'WoT Tank Viewer',
    'doc_url': 'https://kr.cm/f/t/28240/',
    'repo_url': 'https://github.com/Simi4/WoT-Blender-Addons/',
    'category': '3D View',
}

# ruff: noqa: E402

import traceback
from typing import Any

from mathutils import Vector  # type: ignore
import bpy
import bpy.utils.previews  # type: ignore

# local imports
from .settings import *
from .common import *
from .BigWorldModelLoader import *
from .VehicleUtils import *
from .map_viewer import ui as map_viewer_ui
from .havok.loader import load_havok_file


class PluginData:
    def __init__(self):
        self.wot_data_manager = None
        self.current_tank_info = None
        self.current_tank_name = ''
        self.current_vehicle_id = None
        self.wot_version = None
        self.style_items = []
        self.turret_l10n_items = []
        self.hull_l10n_items = []
        self.gun_l10n_items = {}
        self.custom_icons = None

    def load_icons(self):
        self.custom_icons = bpy.utils.previews.new()
        icons_directory = Path(__file__).parent / 'icons'
        self.custom_icons.load('koreanrandom', str(icons_directory / 'koreanrandom.png'), 'IMAGE')

    def free_icons(self):
        bpy.utils.previews.remove(self.custom_icons)

    def load_client_data(self):
        wot_dir_path = addon_prefs().world_of_tanks_game_path
        self.wot_data_manager = WotDataManager.add_client(wot_dir_path)
        return self.wot_data_manager is not None

    def load_tank_info(self, vehicle_id):
        self.current_tank_info = self.wot_data_manager.loadInfo(vehicle_id)


g_PluginData = PluginData()


@disable_undo
def loadModels(col, model_list: list, scene: Any):
    check_tex_in_res_mode = scene.texture_location == 'custom'

    try:
        is_new_collision = scene.tank_status == 'collision_model' and g_PluginData.wot_data_manager.wot_version >= (1, 27, 0)
    except Exception:
        is_new_collision = False

    try:
        new_primitives_format = g_PluginData.wot_data_manager.wot_version >= (0, 9, 12)
    except Exception:
        new_primitives_format = False

    image_cache = {}
    for model in model_list:
        if is_new_collision:
            load_havok_file(g_PluginData.wot_data_manager.res_mgr, col, model)
        else:
            g_BigWorldModelLoader.load(
                g_PluginData.wot_data_manager.res_mgr,
                col,
                model,
                new_primitives_format,
                image_cache,
                Path(addon_prefs().world_of_tanks_custom_path),
                check_tex_in_res_mode,
            )


@disable_undo
def deleteModels():
    if 'Tank Collection' in bpy.data.collections:
        col = bpy.data.collections['Tank Collection']
        for obj in col.objects:
            bpy.data.objects.remove(obj)
        bpy.ops.outliner.orphans_purge(do_recursive=True)


class Apply_Path_Operator(bpy.types.Operator):
    bl_label = 'Operator apply paths'
    bl_idname = 'tv.apply_path_op'

    def execute(self, context):
        if g_PluginData.load_client_data():
            nation_l10n_items = []
            for idx, nation_l10n in g_NationsL10n.INDEX.items():
                nation_l10n_items.append((str(idx), nation_l10n, ''))
            nation_l10n_items.sort()

            category_l10n_items = []
            for idx, category_l10n in g_CategoryL10n.INDEX.items():
                category_l10n_items.insert(idx, (str(idx), category_l10n, ''))

            level_l10n_items = []
            for idx, level_l10n in g_LevelsL10n.INDEX.items():
                level_l10n_items.insert(idx, (str(idx), level_l10n, ''))

            # register
            viewer_register()

            bpy.types.Scene.custom = bpy.props.CollectionProperty(type=CustomProp)
            bpy.types.Scene.custom_index = bpy.props.IntProperty()
            bpy.types.Scene.nations_l10n = bpy.props.EnumProperty(name='Nations', description='', items=nation_l10n_items)
            bpy.types.Scene.category_l10n = bpy.props.EnumProperty(name='Category', description='', items=category_l10n_items)
            bpy.types.Scene.levels_l10n = bpy.props.EnumProperty(name='Tier', description='', items=level_l10n_items)

            # unregister
            startup_unregister()

        else:
            # INFO, WARNING or ERROR
            self.report({'WARNING'}, 'Error in path!')

        return {'FINISHED'}


class Apply_Filter_Operator(bpy.types.Operator):
    bl_label = 'Operator apply filter'
    bl_idname = 'tv.apply_filter'

    def execute(self, context):
        scn = context.scene
        custom = scn.custom
        scn.custom_index = 0
        custom.clear()

        if g_PluginData.wot_data_manager is not None:
            level = int(scn.levels_l10n)
            category = int(scn.category_l10n)
            nation = int(scn.nations_l10n)

            str_exec = 'SELECT rowid, tank_props, tank_name FROM Vehicles'
            str_q = []
            if nation:
                str_q.append('tank_nation=:nation')
            if category:
                str_q.append('tank_type_id=:category')
            if level:
                str_q.append('tank_level=:level')
            if str_q:
                str_exec += ' WHERE ' + ' AND '.join(str_q) + ';'
            for row in g_PluginData.wot_data_manager.sqlite3_cursor.execute(str_exec, {'level': level, 'category': category, 'nation': nation}):
                tank_list_item = custom.add()
                tank_list_item.vehicle_id = row[0]
                tank_list_item.icon_locked = row[1] & 1
                tank_list_item.name = row[2]

        return {'FINISHED'}


def loadByVehicleInfo(scene):
    deleteModels()

    model_list = []

    hull = g_PluginData.current_tank_info.hulls[scene.hulls_l10n]
    chassis = hull.chassis
    splineDesc = chassis.splineDesc
    turret = g_PluginData.current_tank_info.turrets0[scene.turrets_l10n]
    gun = turret.guns[scene.guns_l10n]

    model_status = 'undamaged_model'
    if scene.tank_status in ('undamaged_model', 'destroyed_model', 'collision_model'):
        model_status = scene.tank_status

    # TODO: rework
    def get_path_by_style_status(d):
        if model_status == 'collision_model':
            return d.collision_model
        if scene.tank_style in d.sets:
            return getattr(d.sets[scene.tank_style], model_status)
        return getattr(d.sets['default'], model_status)

    if not scene.use_segment_tracks or model_status != 'undamaged_model' or splineDesc is None:
        model_list.append(
            {
                'File': get_path_by_style_status(chassis),
                'Scale': Vector((1.0, 1.0, 1.0)),
                'Rotation': Vector((0.0, 0.0, 0.0)),
                'Position': Vector((0.0, 0.0, 0.0)),
                'Armors': chassis.armors,
            }
        )

    else:
        model_list.append(
            {
                'File': get_path_by_style_status(chassis),
                'Scale': Vector((1.0, 1.0, 1.0)),
                'Rotation': Vector((0.0, 0.0, 0.0)),
                'Position': Vector((0.0, 0.0, 0.0)),
                'use_segment': True,
            }
        )

        for trackPair in splineDesc.trackPairs:
            segments = trackPair.sets.get(scene.tank_style) or trackPair.sets['default']

            model_list.append(
                {
                    'File': segments.segmentModelLeft,
                    'Scale': Vector((1.0, 1.0, 1.0)),
                    'Rotation': Vector((0.0, 0.0, 0.0)),
                    'Position': Vector((0.0, trackPair.segmentOffset, 0.0)),
                    'is_segment': True,
                    'track_file': trackPair.left,
                    'segmentOffset': trackPair.segmentLength,
                }
            )

            model_list.append(
                {
                    'File': segments.segmentModelRight,
                    'Scale': Vector((1.0, 1.0, 1.0)),
                    'Rotation': Vector((0.0, 0.0, 0.0)),
                    'Position': Vector((0.0, trackPair.segmentOffset, 0.0)),
                    'is_segment': True,
                    'track_file': trackPair.right,
                    'segmentOffset': trackPair.segmentLength,
                }
            )

            if segments.has_two_segments():
                model_list.append(
                    {
                        'File': segments.segment2ModelLeft,
                        'Scale': Vector((1.0, 1.0, 1.0)),
                        'Rotation': Vector((0.0, 0.0, 0.0)),
                        'Position': Vector((0.0, trackPair.segmentLength / 2, 0.0)),
                        'is_segment': True,
                        'track_file': trackPair.left,
                        'segmentOffset': trackPair.segmentLength,
                    }
                )

                model_list.append(
                    {
                        'File': segments.segment2ModelRight,
                        'Scale': Vector((1.0, 1.0, 1.0)),
                        'Rotation': Vector((0.0, 0.0, 0.0)),
                        'Position': Vector((0.0, trackPair.segmentLength / 2, 0.0)),
                        'is_segment': True,
                        'track_file': trackPair.right,
                        'segmentOffset': trackPair.segmentLength,
                    }
                )

    model_list.append(
        {
            'File': get_path_by_style_status(turret),
            'Scale': Vector((1.0, 1.0, 1.0)),
            'Rotation': Vector((0.0, 0.0, 0.0)),
            'Position': hull.turretPositions,
            'Armors': turret.armors,
        }
    )

    model_list.append(
        {
            'File': get_path_by_style_status(hull),
            'Scale': Vector((1.0, 1.0, 1.0)),
            'Rotation': Vector((0.0, 0.0, 0.0)),
            'Position': chassis.hullPosition,
            'Armors': hull.armors,
        }
    )

    model_list.append(
        {
            'File': get_path_by_style_status(gun),
            'Scale': Vector((1.0, 1.0, 1.0)),
            'Rotation': Vector((0.0, 0.0, 0.0)),
            'Position': turret.gunPosition + hull.turretPositions,
            'Armors': gun.armors,
        }
    )

    if 'Tank Collection' not in bpy.data.collections:
        col = bpy.data.collections.new('Tank Collection')
        scene.collection.children.link(col)
    else:
        col = bpy.data.collections['Tank Collection']

    loadModels(col, model_list, scene)


class Apply_Vehicle_Operator(bpy.types.Operator):
    bl_label = 'Operator apply vehicle'
    bl_idname = 'tv.apply_vehicle'

    @classmethod
    def poll(self, context):
        if len(context.scene.custom):
            custom = context.scene.custom[context.scene.custom_index]
            if (g_PluginData.current_vehicle_id != custom.vehicle_id) and ((context.active_object is None) or (context.active_object.mode == 'OBJECT')):
                return True
        return False

    def execute(self, context):
        if g_PluginData.current_tank_name == '':
            registerModelSettings()

        scn = context.scene
        custom = scn.custom[scn.custom_index]
        g_PluginData.current_tank_name = custom.name
        g_PluginData.current_vehicle_id = custom.vehicle_id

        scn.use_segment_tracks = addon_prefs().force_use_segment_tracks
        scn.tank_status = 'undamaged_model'

        g_PluginData.load_tank_info(custom.vehicle_id)
        g_PluginData.turret_l10n_items = t_items = []
        g_PluginData.gun_l10n_items.clear()
        for turret_name, turret_info in g_PluginData.current_tank_info.turrets0.items():
            t_items.append((turret_name, turret_name, ''))
            gun_list = []
            for gun_name, gun_info in turret_info.guns.items():
                gun_list.append((gun_name, gun_name, ''))
            g_PluginData.gun_l10n_items[turret_name] = gun_list

        try:
            t_items.sort()
            scn.turrets_l10n = t_items[0][0]
            g_PluginData.gun_l10n_items[t_items[0][0]].sort()
            scn.guns_l10n = g_PluginData.gun_l10n_items[t_items[0][0]][0][0]
        except Exception:
            traceback.print_exc()

        g_PluginData.hull_l10n_items = []
        for hull_name in g_PluginData.current_tank_info.hulls.keys():
            g_PluginData.hull_l10n_items.append((hull_name, hull_name, ''))
        scn.hulls_l10n = g_PluginData.hull_l10n_items[0][0]

        g_PluginData.style_items = [('default', 'default', '')]
        str_exec = 'SELECT style_name, models_set FROM Styles WHERE vehicle_id=?'
        for row in g_PluginData.wot_data_manager.sqlite3_cursor.execute(str_exec, (custom.vehicle_id,)):
            g_PluginData.style_items.append((row[1], row[0], ''))
        scn.tank_style = g_PluginData.style_items[0][0]

        logger.info('tank name: `%s`' % g_PluginData.current_tank_name)
        try:
            loadByVehicleInfo(scn)
        except Exception:
            self.report({'ERROR'}, 'Error in load `%s`!' % g_PluginData.current_tank_name)
            traceback.print_exc()

        return {'FINISHED'}


class Reopen_Vehicle_Operator(bpy.types.Operator):
    bl_label = 'Operator reopen vehicle'
    bl_idname = 'tv.reopen_vehicle'

    @classmethod
    def poll(self, context):
        if (context.active_object is None) or (context.active_object.mode == 'OBJECT'):
            return True
        return False

    def execute(self, context):
        logger.info('reopen tank')
        try:
            loadByVehicleInfo(context.scene)
        except Exception:
            self.report({'ERROR'}, 'Error in reopen `%s`!' % g_PluginData.current_tank_name)
            traceback.print_exc()

        return {'FINISHED'}


def turret_l10n_items_callback(scene, context):
    return g_PluginData.turret_l10n_items


def hull_l10n_items_callback(scene, context):
    return g_PluginData.hull_l10n_items


def gun_l10n_items_callback(scene, context):
    if context is not None:
        return g_PluginData.gun_l10n_items.get(context.scene.turrets_l10n)
    return []


def tank_style_items_callback(scene, context):
    return g_PluginData.style_items


def registerModelSettings():
    vehicle_settings_register()

    bpy.types.Scene.turrets_l10n = bpy.props.EnumProperty(
        name='Turret',
        description='',
        items=turret_l10n_items_callback,
    )
    bpy.types.Scene.hulls_l10n = bpy.props.EnumProperty(
        name='Hull',
        description='',
        items=hull_l10n_items_callback,
    )
    bpy.types.Scene.guns_l10n = bpy.props.EnumProperty(
        name='Gun',
        description='',
        items=gun_l10n_items_callback,
    )
    bpy.types.Scene.use_segment_tracks = bpy.props.BoolProperty(
        name='Use segment tracks (Experimental)',
        description='',
        default=False,
    )
    bpy.types.Scene.tank_style = bpy.props.EnumProperty(
        name='Tank style',
        description='',
        items=tank_style_items_callback,
    )
    bpy.types.Scene.tank_status = bpy.props.EnumProperty(
        name='Tank Mesh',
        description='',
        items=(
            ('undamaged_model', 'Undamaged model', ''),
            ('destroyed_model', 'Destroyed model', ''),
            ('collision_model', 'Collision model', ''),
        ),
    )
    bpy.types.Scene.texture_location = bpy.props.EnumProperty(
        name='Texture Location',
        description='',
        items=(
            ('custom', 'Also res_mods folder', ''),
            ('packages', 'Pkg Files', ''),
        ),
    )


#####################################################################
# Panel TankViewer Settings


class PANEL_PT_TankViewer_ModelSettings(bpy.types.Panel):
    bl_label = 'Tank Options'
    bl_space_type = 'VIEW_3D'
    bl_region_type = REGION_TYPE
    bl_category = CATEGORY
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scn = context.scene

        layout.prop(scn, 'tank_style')
        layout.separator()

        box = layout.row().box()
        box.prop(scn, 'hulls_l10n')
        box.prop(scn, 'turrets_l10n')
        box.prop(scn, 'guns_l10n')
        layout.separator()

        layout.prop(scn, 'use_segment_tracks')
        layout.separator()

        box = layout.row().box()
        box.prop(scn, 'tank_status')
        layout.separator()

        box = layout.row().box()
        box.prop(scn, 'texture_location')
        layout.separator()

        layout.operator('tv.reopen_vehicle', text='Reload model')


class PANEL_PT_TankViewer_Start(bpy.types.Panel):
    bl_label = 'Start'
    bl_space_type = 'VIEW_3D'
    bl_region_type = REGION_TYPE
    bl_category = CATEGORY

    def draw(self, context):
        layout = self.layout

        layout.operator('tv.apply_path_op', text='Parse WoT')


class PANEL_PT_TankFilter(bpy.types.Panel):
    bl_label = 'Tank Filter'
    bl_space_type = 'VIEW_3D'
    bl_region_type = REGION_TYPE
    bl_category = CATEGORY

    def draw(self, context):
        layout = self.layout

        scn = context.scene

        box = layout.row().box()
        box.prop(scn, 'nations_l10n', text='')
        box.prop(scn, 'category_l10n', text='')
        box.prop(scn, 'levels_l10n', text='')

        layout.separator()
        layout.operator('tv.apply_filter', text='Apply Filter')


class CustomProp(bpy.types.PropertyGroup):
    vehicle_id: bpy.props.IntProperty()  # type: ignore
    name: bpy.props.StringProperty()  # type: ignore
    icon_locked: bpy.props.BoolProperty()  # type: ignore


class UI_UL_Tank_List(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if not item.icon_locked:
            layout.label(text=item.name, translate=False)

        else:
            layout.label(text=item.name, icon='LOCKED', translate=False)


class PANEL_PT_TankList(bpy.types.Panel):
    bl_label = 'Tank List'
    bl_space_type = 'VIEW_3D'
    bl_region_type = REGION_TYPE
    bl_category = CATEGORY

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        layout.row().template_list('UI_UL_Tank_List', '', scn, 'custom', scn, 'custom_index')

        layout.separator()
        layout.operator('tv.apply_vehicle', text='Load Tank')


class TankViewerPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_PREF_ID

    world_of_tanks_game_path: bpy.props.StringProperty(
        name='WoT Path',
        subtype='DIR_PATH',
        default=WOT_PATH_DEFAULT,
    )  # type: ignore

    world_of_tanks_custom_path: bpy.props.StringProperty(
        name='Custom Path',
        subtype='DIR_PATH',
        default=WOT_CUSTOM_PATH_DEFAULT,
    )  # type: ignore

    force_use_segment_tracks: bpy.props.BoolProperty(
        name='Force use segment tracks (Experimental)',
        default=False,
    )  # type: ignore

    map_viewer_load_normals: bpy.props.BoolProperty(
        name='Map Viewer (Experimental Load Terrain Normals)',
        default=False,
    )  # type: ignore

    map_viewer_load_wetness: bpy.props.BoolProperty(
        name='Map Viewer (Experimental Load Terrain Wetness)',
        default=False,
    )  # type: ignore

    map_viewer_load_objects: bpy.props.BoolProperty(
        name='Map Viewer (Experimental Load Objects)',
        default=False,
    )  # type: ignore

    def draw(self, context):
        layout = self.layout

        layout.prop(self, 'world_of_tanks_game_path')
        layout.prop(self, 'world_of_tanks_custom_path')
        layout.prop(self, 'force_use_segment_tracks')

        layout.separator()

        box = layout.row().box()
        box.prop(self, 'map_viewer_load_normals')
        box.prop(self, 'map_viewer_load_wetness')
        box.prop(self, 'map_viewer_load_objects')


def menu_func_tankviewer(self, context):
    self.layout.menu('INFO_MT_tankviewer')


class INFO_MT_tankviewer(bpy.types.Menu):
    bl_label = 'About TankViewer'

    def draw(self, context):
        layout = self.layout

        layout.label(text='Version: %d.%d.%d' % bl_info['version'][:])
        layout.label(text='Author: %s' % bl_info['author'])

        layout.separator()

        layout.operator('wm.url_open', text='TankViewer Website', icon_value=g_PluginData.custom_icons['koreanrandom'].icon_id).url = bl_info['doc_url']
        layout.operator('wm.url_open', text='TankViewer Repository', icon='URL').url = bl_info['repo_url']


generic_register, generic_unregister = bpy.utils.register_classes_factory(
    (
        TankViewerPreferences,
        INFO_MT_tankviewer,
    )
)


startup_register, startup_unregister = bpy.utils.register_classes_factory(
    (
        PANEL_PT_TankViewer_Start,
        Apply_Path_Operator,
    )
)


viewer_register, viewer_unregister = bpy.utils.register_classes_factory(
    (
        CustomProp,
        Apply_Filter_Operator,
        Apply_Vehicle_Operator,
        PANEL_PT_TankFilter,
        PANEL_PT_TankList,
        UI_UL_Tank_List,
    )
)


vehicle_settings_register, vehicle_settings_unregister = bpy.utils.register_classes_factory(
    (
        PANEL_PT_TankViewer_ModelSettings,
        Reopen_Vehicle_Operator,
    )
)


def register():
    global g_PluginData

    generic_register()
    startup_register()

    g_PluginData.load_icons()
    bpy.types.TOPBAR_MT_help.prepend(menu_func_tankviewer)
    bpy.app.translations.register(__package__, translations_dict)
    map_viewer_ui.register()


def unregister():
    global g_PluginData

    bpy.types.TOPBAR_MT_help.remove(menu_func_tankviewer)
    generic_unregister()

    bpy.app.translations.unregister(__package__)

    if g_PluginData.wot_data_manager is not None:
        if g_PluginData.current_tank_name:
            vehicle_settings_unregister()
            del bpy.types.Scene.hulls_l10n
            del bpy.types.Scene.turrets_l10n
            del bpy.types.Scene.guns_l10n
            del bpy.types.Scene.use_segment_tracks
            del bpy.types.Scene.tank_style
            del bpy.types.Scene.tank_status
        bpy.context.scene.custom.clear()
        del bpy.types.Scene.custom
        del bpy.types.Scene.custom_index
        del bpy.types.Scene.nations_l10n
        del bpy.types.Scene.category_l10n
        del bpy.types.Scene.levels_l10n
        viewer_unregister()

    else:
        startup_unregister()

    g_PluginData.free_icons()
    g_PluginData = PluginData()
    map_viewer_ui.unregister()
