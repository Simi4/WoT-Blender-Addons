
# blender imports
import bpy  # type: ignore

# local imports
from ..common import *
from ..settings import *
from . import utils
from . import loader


wot_parse_status = False
current_map_info = None


@disable_undo
def delete_map():
    if col := bpy.data.collections.get('Map Collection'):
        bpy.data.collections.remove(col)
        bpy.ops.outliner.orphans_purge(do_recursive=True)


class Apply_Path_Operator(bpy.types.Operator):
    bl_label = 'Operator apply paths'
    bl_idname = 'wotmapsviewer.apply_path_op'

    def execute(self, context):
        global wot_parse_status
        wot_path = Path(addon_prefs().world_of_tanks_game_path)
        version_xml_path = wot_path / 'version.xml'
        if not version_xml_path.is_file():
            # INFO, WARNING or ERROR
            self.report({'WARNING'}, 'Error in path!')
            return {'FINISHED'}

        # register
        bpy.utils.register_class(UI_UL_WoTMaps_List)
        bpy.utils.register_class(MapsCustomProp)
        bpy.utils.register_class(Show_Map_Info_Operator)
        bpy.types.Scene.wot_maps_list_custom = bpy.props.CollectionProperty(type=MapsCustomProp)
        bpy.types.Scene.wot_maps_list_custom_index = bpy.props.IntProperty()
        bpy.utils.register_class(PANEL_PT_WoTMaps_List)

        # unregister
        bpy.utils.unregister_class(PANEL_PT_WoTMapsViewer_Start)
        bpy.utils.unregister_class(Apply_Path_Operator)

        utils.init(wot_path / 'res', version_xml_path)
        maps_list = utils.load_maps_dictionary()

        scn = context.scene
        scn.wot_maps_list_custom_index = 0
        scn.wot_maps_list_custom.clear()

        for map_tuple in maps_list:
            wot_maps_list_item = scn.wot_maps_list_custom.add()
            wot_maps_list_item.name_l10n = map_tuple[2]
            wot_maps_list_item.map_name = map_tuple[1]
            wot_maps_list_item.map_id = map_tuple[0]

        wot_parse_status = True
        return {'FINISHED'}


class PANEL_PT_WoTMapsViewer_Start(bpy.types.Panel):
    bl_label = 'Start'
    bl_space_type = 'VIEW_3D'
    bl_region_type = MAP_VIEWER_REGION_TYPE
    bl_category = MAP_VIEWER_CATEGORY

    def draw(self, context):
        layout = self.layout

        layout.operator('wotmapsviewer.apply_path_op', text='Parse WoT')


class MapsCustomProp(bpy.types.PropertyGroup):
    name_l10n: bpy.props.StringProperty()  # type: ignore
    map_name: bpy.props.StringProperty()  # type: ignore
    map_id: bpy.props.IntProperty()  # type: ignore


class UI_UL_WoTMaps_List(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.name_l10n, translate=False)


class PANEL_PT_WoTMaps_List(bpy.types.Panel):
    bl_label = 'WoT Map List'
    bl_space_type = 'VIEW_3D'
    bl_region_type = MAP_VIEWER_REGION_TYPE
    bl_category = MAP_VIEWER_CATEGORY

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        layout.row().template_list(
            'UI_UL_WoTMaps_List', '', scn, 'wot_maps_list_custom', scn, 'wot_maps_list_custom_index'
        )

        layout.separator()
        layout.operator('wotmapsviewer.show_map_info', text='Show Map')

        if current_map_info:
            layout.label(text=str(current_map_info['geometry']))


class Show_Map_Info_Operator(bpy.types.Operator):
    bl_label = 'Operator show map info paths'
    bl_idname = 'wotmapsviewer.show_map_info'

    def execute(self, context):
        global current_map_info
        scn = context.scene
        map_name = scn.wot_maps_list_custom[scn.wot_maps_list_custom_index].map_name
        current_map_info = utils.load_map_info(map_name)

        delete_map()

        res_pkg_path = Path(addon_prefs().world_of_tanks_game_path) / 'res' / 'packages'

        SHOW_STAT = False
        if SHOW_STAT:
            import cProfile
            import pstats
            from pstats import SortKey

            pr = cProfile.Profile()
            pr.enable()

        loader.load(scn, current_map_info, res_pkg_path)

        if SHOW_STAT:
            pr.disable()
            st = pstats.Stats(pr).strip_dirs().sort_stats(SortKey.CUMULATIVE)
            st.dump_stats('loader.prof')
            st.print_stats(20)

        return {'FINISHED'}


def unregister():
    global wot_parse_status, current_map_info
    current_map_info = None

    if wot_parse_status:
        wot_parse_status = False
        bpy.utils.unregister_class(UI_UL_WoTMaps_List)
        bpy.utils.unregister_class(MapsCustomProp)
        bpy.utils.unregister_class(Show_Map_Info_Operator)
        bpy.utils.unregister_class(PANEL_PT_WoTMaps_List)
        del bpy.types.Scene.wot_maps_list_custom
        del bpy.types.Scene.wot_maps_list_custom_index
        utils.fini()

    else:
        bpy.utils.unregister_class(PANEL_PT_WoTMapsViewer_Start)
        bpy.utils.unregister_class(Apply_Path_Operator)


def register():
    bpy.utils.register_class(PANEL_PT_WoTMapsViewer_Start)
    bpy.utils.register_class(Apply_Path_Operator)
