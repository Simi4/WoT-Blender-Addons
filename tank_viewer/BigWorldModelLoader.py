
__all__ = ('g_BigWorldModelLoader',)


#####################################################################
# imports

import os

from .LoadDataMesh import LoadDataMesh
from .BigWorldMaterial import *
from .common import *

import bpy

from mathutils import Vector  # type: ignore
from bpy_extras.image_utils import load_image  # type: ignore
from bpy_extras.io_utils import unpack_list  # type: ignore


#####################################################################
# BigWorldModelLoader


class BigWorldModelLoader:
    def curve_track(self, col, track_elem):
        cu_positions = []

        visual_scene = track_elem.find('library_visual_scenes/visual_scene')

        to_invert = False

        if track_elem.find('asset') is not None:
            cu_name = '%s_track' % os.path.splitext(visual_scene.find('name').text.strip())[0]

            meter = float(track_elem.find('asset/unit/meter').text)
            root_matrix = meter * tv_AsMatrix4x4T(visual_scene.find('node/matrix').text)

            for node in visual_scene.findall('node/node'):
                matrix4x4 = tv_AsMatrix4x4T(node.find('matrix').text) @ root_matrix
                cu_positions.append(Vector((matrix4x4[3].x, -matrix4x4[3].z, matrix4x4[3].y, 0.01)))

            dif_pos = cu_positions[0] - cu_positions[1]
            if dif_pos.y < 0:
                to_invert = True

            elif dif_pos.z < 0:
                to_invert = True

        else:
            cu_name = '%s_track' % os.path.splitext(visual_scene.find('node/name').text.strip())[0]
            for node in visual_scene.findall('node/node'):
                vector3 = StrToVector(node.find('position').text).xzy
                cu_positions.append(Vector((vector3.x, vector3.y, vector3.z, 0.01)))

        if bpy.data.curves.get(cu_name) and bpy.data.objects.get(cu_name):
            return bpy.data.curves[cu_name], bpy.data.objects[cu_name], to_invert

        cu = bpy.data.curves.new(cu_name, 'CURVE')
        cu.dimensions = '3D'

        nu = cu.splines.new('NURBS')
        nu.points.add(len(cu_positions) - 1)
        nu.points.foreach_set('co', unpack_list(cu_positions))
        nu.use_cyclic_u = True
        nu.order_u = 2

        cu_ob = bpy.data.objects.new(cu_name, cu)
        col.objects.link(cu_ob)

        return cu, cu_ob, to_invert

    def load(self, res_mgr, col, model, new_ext_format: bool, image_cache: dict, custom_res_path: Path, check_tex_in_res_mode: bool):
        model_name = os.path.splitext(os.path.basename(model['File']))[0]
        logger.info('Start loading model: `%s`' % model['File'])
        unp_model_file = res_mgr.open_file(model['File'])

        if not unp_model_file:
            return

        model_xml = g_XmlUnpacker.read(unp_model_file)
        del unp_model_file

        if model_xml.find('nodefullVisual') is not None:
            visual_filepath = tv_AsNormPath(model_xml.find('nodefullVisual').text)
            visual_filepath = '%s.visual' % visual_filepath
        elif model_xml.find('nodelessVisual') is not None:
            visual_filepath = tv_AsNormPath(model_xml.find('nodelessVisual').text)
            visual_filepath = '%s.visual' % visual_filepath
        else:
            logger.warning('`nodefullVisual` and `nodelessVisual` not in .model')
            visual_filepath = '%s.visual' % os.path.splitext(model['File'])[0]

        del model_xml

        if new_ext_format:
            visual_filepath += '_processed'

        unp_visual_file = res_mgr.open_file(visual_filepath)

        if not unp_visual_file:
            logger.error('Error in visual unpacking')
            return

        visual_xml = g_XmlUnpacker.read(unp_visual_file)
        del unp_visual_file

        if visual_xml.find('primitivesName') is not None:
            primitives_filepath = '%s.primitives' % visual_xml.find('primitivesName').text.strip()
            if new_ext_format:
                primitives_filepath += '_processed'
        else:
            primitives_filepath = visual_filepath.replace('.visual', '.primitives')

        unp_primitives_file = res_mgr.open_file(primitives_filepath)

        if visual_xml.find('renderSet') is None:
            logger.warning('renderSet is None')
            return

        for renderSet in visual_xml.findall('renderSet'):
            vres_name = renderSet.find('geometry/vertices').text.strip()
            pres_name = renderSet.find('geometry/primitive').text.strip()

            if len(vres_name.split('.')) > 1:
                mesh_name = bpy.path.clean_name(os.path.splitext(vres_name)[0])
            else:
                mesh_name = bpy.path.clean_name(model_name)

            if model.get('use_segment'):
                if mesh_name.startswith(('track_L', 'track_R', 'pasted__R', 'polySurfaceShape', 'tmp_trackShape', 'pasted__track_', 'lod0chassis01tracks')):
                    continue

            bmesh = bpy.data.meshes.new(mesh_name)

            uv2_name = ''
            if renderSet.find('geometry/stream') is not None:
                stream_res_name = renderSet.find('geometry/stream').text.strip()
                if 'uv2' in stream_res_name:
                    uv2_name = stream_res_name

            dataMesh = LoadDataMesh(unp_primitives_file, vres_name, pres_name, uv2_name, new_ext_format)

            bmesh.vertices.add(len(dataMesh.vertices))
            bmesh.vertices.foreach_set('co', unpack_list(dataMesh.vertices))

            # TODO:
            # bmesh.vertices.foreach_set('normal', unpack_list(dataMesh.normal_list))

            nbr_faces = len(dataMesh.indices)
            bmesh.polygons.add(nbr_faces)

            bmesh.polygons.foreach_set('loop_start', range(0, nbr_faces * 3, 3))
            bmesh.polygons.foreach_set('loop_total', (3,) * nbr_faces)

            bmesh.loops.add(nbr_faces * 3)
            bmesh.loops.foreach_set('vertex_index', unpack_list(dataMesh.indices))

            nbr_faces = len(bmesh.polygons)
            bmesh.polygons.foreach_set('use_smooth', [True] * nbr_faces)

            uv2_faces = None
            if uv2_name:
                if dataMesh.uv2_list is not None:
                    uv2_faces = bmesh.uv_layers.new()
                    uv2_name = bpy.path.clean_name(uv2_name)
                    uv2_faces.name = uv2_name
                else:
                    uv2_name = ''

            if dataMesh.uv_list is not None:
                uv_faces = bmesh.uv_layers.new()
                uv_faces.name = 'uv1'
                uv_faces.active = True

                uv_layer = bmesh.uv_layers['uv1'].data[:]
                uv2_layer = bmesh.uv_layers[uv2_name].data[:] if uv2_faces else None

                for poly in bmesh.polygons:
                    for li in poly.loop_indices:
                        vi = bmesh.loops[li].vertex_index
                        uv_layer[li].uv = dataMesh.uv_list[vi]
                        if uv2_name:
                            uv2_layer[li].uv = dataMesh.uv2_list[vi]
            else:
                logger.warning('uv_faces is None')

            for primitiveGroup in renderSet.findall('geometry/primitiveGroup'):
                _identifier = primitiveGroup.findtext('material/identifier').strip()
                _index = AsInt(primitiveGroup.text)

                if primitiveGroup.find('material/fx') is not None:
                    _shader_name = os.path.basename(primitiveGroup.findtext('material/fx').strip())

                    propertiesGroup = dict()
                    for prop in primitiveGroup.findall('material/property'):
                        prop_type = prop.text.strip()

                        match prop_type:
                            case 'g_useNormalPackDXT1' | 'g_useDetailMetallic' | 'g_defaultPBSConversionParams' | 'alphaTestEnable' | 'doubleSided':
                                Bool_ = tv_AsBool(prop.findtext('Bool'))
                                propertiesGroup[prop_type] = {'Bool': Bool_}

                            case (
                                'g_detailUVTiling'
                                | 'g_crashUVTiling'
                                | 'g_albedoConversions'
                                | 'g_glossConversions'
                                | 'g_metallicConversions'
                                | 'g_albedoCorrection'
                            ):
                                Vector4_ = StrToVector(prop.findtext('Vector4'))
                                propertiesGroup[prop_type] = {'Vector4': Vector4_}

                            case 'g_detailPower' | 'g_detailPowerAlbedo' | 'g_detailPowerGloss' | 'g_maskBias' | 'crash_coefficient':
                                Float_ = float(prop.findtext('Float'))
                                propertiesGroup[prop_type] = {'Float': Float_}

                            case 'alphaReference':
                                Int_ = AsInt(prop.findtext('Int'))
                                propertiesGroup[prop_type] = {'Int': Int_}

                            case 'excludeMaskAndAOMap' | 'crashTileMap':
                                pass

                            case 'diffuseMap' | 'diffuseMap2' | 'normalMap' | 'specularMap' | 'metallicDetailMap' | 'metallicGlossMap':
                                sd_in_pkg_path = tv_AsNormPath(prop.findtext('Texture').lower())

                                sd_in_pkg_path_we, _ext = os.path.splitext(sd_in_pkg_path)
                                hd_in_pkg_path = f'{sd_in_pkg_path_we}_hd{_ext}'

                                img_path = None
                                if check_tex_in_res_mode:
                                    img_list = (custom_res_path / hd_in_pkg_path, custom_res_path / sd_in_pkg_path)
                                    for img_p in img_list:
                                        if img_p.is_file():
                                            img_path = str(img_p)
                                            break

                                if img_path is not None:
                                    if not (new_image := image_cache.get(img_path)):
                                        new_image = load_image(img_path)
                                        image_cache[texture_in_pkg_path] = new_image
                                else:
                                    if res_mgr.exists(hd_in_pkg_path):
                                        texture_in_pkg_path = hd_in_pkg_path
                                    elif res_mgr.exists(sd_in_pkg_path):
                                        # fallback to sd
                                        texture_in_pkg_path = sd_in_pkg_path
                                    else:
                                        continue

                                    if not (new_image := image_cache.get(texture_in_pkg_path)):
                                        with res_mgr.open_file(texture_in_pkg_path) as f:
                                            data = f.read()
                                        new_image = load_image_from_memory(data, texture_in_pkg_path)
                                        image_cache[texture_in_pkg_path] = new_image

                                propertiesGroup[prop_type] = {'Texture': new_image}

                            case _:
                                logger.warning(f'unknown {prop_type=}')

                    material = bpy.data.materials.new(_identifier)
                    setBigWorldMaterial(_shader_name, material, uv2_name, propertiesGroup)

                    bmesh.materials.append(material)

                    startIndex = dataMesh.PrimitiveGroups[_index]['startIndex'] // 3
                    endPrimitives = startIndex + dataMesh.PrimitiveGroups[_index]['nPrimitives']

                    for fidx, pl in enumerate(bmesh.polygons):
                        if startIndex <= fidx <= endPrimitives:
                            pl.material_index = _index

            bmesh.validate()
            bmesh.update()

            ob = bpy.data.objects.new(mesh_name, bmesh)
            col.objects.link(ob)

            ob.scale = model['Scale']
            ob.rotation_euler = model['Rotation']
            ob.location = model['Position']

            if model.get('is_segment'):
                unp_track_file = res_mgr.open_file(model['track_file'])

                if unp_track_file:
                    track_xml = g_XmlUnpacker.read(unp_track_file)
                    del unp_track_file

                    cu, cu_ob, to_invert = self.curve_track(col, track_xml)

                    from math import pi

                    ob.rotation_euler[2] += pi
                    if to_invert:
                        ob.rotation_euler[1] += pi

                    ob_mod_Array = ob.modifiers.new('Array', 'ARRAY')
                    ob_mod_Array.fit_type = 'FIT_CURVE'
                    ob_mod_Array.curve = cu_ob

                    ob_mod_Array.use_relative_offset = False
                    ob_mod_Array.use_constant_offset = True

                    ob_mod_Array.constant_offset_displace[0] = 0
                    ob_mod_Array.constant_offset_displace[1] = model['segmentOffset']

                    ob_mod_Curve = ob.modifiers.new('Curve', 'CURVE')
                    ob_mod_Curve.object = cu_ob
                    ob_mod_Curve.deform_axis = 'NEG_Y'

        del unp_primitives_file


g_BigWorldModelLoader = BigWorldModelLoader()
