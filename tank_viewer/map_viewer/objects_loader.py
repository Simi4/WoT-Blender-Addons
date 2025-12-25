
# imports
from typing import Any

# blender imports
import bpy  # type: ignore
from mathutils import Matrix, Vector  # type: ignore

# local imports
from ..common import logger
from .atlas_loader import load_atlas, load_atlas_dds
from .compiled_space.universal_space import UniversalMesh
from .LoadDataMesh_v2 import LoadDataMesh_v2
from .loader_context import LoaderContext
from .terrain_loader import get_image


# xzy to xyz
flip_mat = Matrix(
    (
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    )
)


def add_node(node_tree: bpy.types.NodeTree, node_name: str, loc: tuple[float, float]) -> bpy.types.ShaderNode:
    node = node_tree.nodes.new(node_name)
    node.hide = True
    node.location = (loc[0] * 100.0, loc[1] * 100.0)
    return node


def load_pbs_tiled_atlas_material(node_tree: bpy.types.NodeTree, diff_node: bpy.types.ShaderNodeBsdfDiffuse, props: dict[str, Any], ctx: LoaderContext):
    atlasAlbedoHeight = props.get('atlasAlbedoHeight')
    if atlasAlbedoHeight is None:
        return

    g_atlasIndexes = props.get('g_atlasIndexes')
    if g_atlasIndexes is None:
        return

    g_atlasSizes = props.get('g_atlasSizes')
    if g_atlasSizes is None:
        return

    atlasBlend = props.get('atlasBlend')
    if atlasBlend is None:
        return

    # atlasBlend is always png
    if not atlasBlend.endswith('.png'):
        return

    if atlasAlbedoHeight.endswith('.atlas'):
        atlas = load_atlas(ctx, atlasAlbedoHeight + '_processed')
    else:
        atlas = load_atlas_dds(atlasAlbedoHeight, g_atlasIndexes)

    if not atlas:
        return

    g_tile0Tint = Vector(props.get('g_tile0Tint', (1, 1, 1, 1))[:3])
    g_tile1Tint = Vector(props.get('g_tile1Tint', (1, 1, 1, 1))[:3])
    g_tile2Tint = Vector(props.get('g_tile2Tint', (1, 1, 1, 1))[:3])

    uv2_node = add_node(node_tree, 'ShaderNodeUVMap', (-12, 0))
    uv2_node.uv_map = 'uv2'

    uv1_node = add_node(node_tree, 'ShaderNodeUVMap', (-12, -2))
    uv1_node.uv_map = 'uv1'

    wrap_math_node = add_node(node_tree, 'ShaderNodeVectorMath', (-10, -2))
    wrap_math_node.operation = 'WRAP'
    node_tree.links.new(uv1_node.outputs['UV'], wrap_math_node.inputs[0])
    wrap_math_node.inputs[1].default_value = Vector((0.9375, 0.9375, 0.0))
    wrap_math_node.inputs[2].default_value = Vector((0.0625, 0.0625, 0.0))

    mul_add_math_node = add_node(node_tree, 'ShaderNodeVectorMath', (-10, 0))
    mul_add_math_node.operation = 'MULTIPLY_ADD'
    multipler = Vector((1.0 / g_atlasSizes[2], 1.0 / g_atlasSizes[3], 0.0))
    mul_add_math_node.inputs[1].default_value = multipler
    atlas_scale = Vector((g_atlasIndexes[3] % g_atlasSizes[2], g_atlasSizes[3] - 1.0 - g_atlasIndexes[3] // g_atlasSizes[2], 0.0)) * multipler
    mul_add_math_node.inputs[2].default_value = atlas_scale
    node_tree.links.new(uv2_node.outputs['UV'], mul_add_math_node.inputs[0])

    atlasBlend = atlasBlend[:-4] + '.dds'
    atlasBlend_img = get_image(ctx, atlasBlend)
    atlasBlend_img.colorspace_settings.name = 'Non-Color'
    atlasBlend_tex_node = add_node(node_tree, 'ShaderNodeTexImage', (-8, 0))
    atlasBlend_tex_node.image = atlasBlend_img
    node_tree.links.new(mul_add_math_node.outputs[0], atlasBlend_tex_node.inputs['Vector'])

    sep_col_node = add_node(node_tree, 'ShaderNodeSeparateColor', (-2, 0))
    node_tree.links.new(atlasBlend_tex_node.outputs['Color'], sep_col_node.inputs['Color'])

    sub1_math_node = add_node(node_tree, 'ShaderNodeMath', (-2, -0.5))
    sub1_math_node.operation = 'SUBTRACT'
    sub1_math_node.inputs[0].default_value = 1.0
    node_tree.links.new(sep_col_node.outputs['Red'], sub1_math_node.inputs[1])

    sub2_math_node = add_node(node_tree, 'ShaderNodeMath', (-2, -1))
    sub2_math_node.operation = 'SUBTRACT'
    sub2_math_node.use_clamp = True
    node_tree.links.new(sub1_math_node.outputs['Value'], sub2_math_node.inputs[0])
    node_tree.links.new(sep_col_node.outputs['Green'], sub2_math_node.inputs[1])

    am0_img = get_image(ctx, atlas[int(g_atlasIndexes[0])].path)
    am0_img.alpha_mode = 'PREMUL'
    am0_tex_node = add_node(node_tree, 'ShaderNodeTexImage', (-8, -1))
    am0_tex_node.image = am0_img
    node_tree.links.new(wrap_math_node.outputs[0], am0_tex_node.inputs['Vector'])

    am0_mul_math_node = add_node(node_tree, 'ShaderNodeVectorMath', (-5, -1))
    am0_mul_math_node.operation = 'MULTIPLY'
    am0_mul_math_node.inputs[1].default_value = g_tile0Tint
    node_tree.links.new(am0_tex_node.outputs['Color'], am0_mul_math_node.inputs[0])

    am1_img = get_image(ctx, atlas[int(g_atlasIndexes[1])].path)
    am1_img.alpha_mode = 'PREMUL'
    am1_tex_node = add_node(node_tree, 'ShaderNodeTexImage', (-8, -1.5))
    am1_tex_node.image = am1_img
    node_tree.links.new(wrap_math_node.outputs[0], am1_tex_node.inputs['Vector'])

    am1_mul_math_node = add_node(node_tree, 'ShaderNodeVectorMath', (-5, -1.5))
    am1_mul_math_node.operation = 'MULTIPLY'
    am1_mul_math_node.inputs[1].default_value = g_tile1Tint
    node_tree.links.new(am1_tex_node.outputs['Color'], am1_mul_math_node.inputs[0])

    am2_img = get_image(ctx, atlas[int(g_atlasIndexes[2])].path)
    am2_img.alpha_mode = 'PREMUL'
    am2_tex_node = add_node(node_tree, 'ShaderNodeTexImage', (-8, -2))
    am2_tex_node.image = am2_img
    node_tree.links.new(wrap_math_node.outputs[0], am2_tex_node.inputs['Vector'])

    am2_mul_math_node = add_node(node_tree, 'ShaderNodeVectorMath', (-5, -2))
    am2_mul_math_node.operation = 'MULTIPLY'
    am2_mul_math_node.inputs[1].default_value = g_tile2Tint
    node_tree.links.new(am2_tex_node.outputs['Color'], am2_mul_math_node.inputs[0])

    mix0_math_node = add_node(node_tree, 'ShaderNodeVectorMath', (0, 0))
    mix0_math_node.operation = 'MULTIPLY_ADD'
    node_tree.links.new(am0_mul_math_node.outputs[0], mix0_math_node.inputs[0])
    node_tree.links.new(sep_col_node.outputs['Red'], mix0_math_node.inputs[1])
    node_tree.links.new(mix0_math_node.outputs[0], diff_node.inputs['Color'])

    mix1_math_node = add_node(node_tree, 'ShaderNodeVectorMath', (0, -0.5))
    mix1_math_node.operation = 'MULTIPLY_ADD'
    node_tree.links.new(am1_mul_math_node.outputs[0], mix1_math_node.inputs[0])
    node_tree.links.new(sep_col_node.outputs['Green'], mix1_math_node.inputs[1])
    node_tree.links.new(mix1_math_node.outputs[0], mix0_math_node.inputs[2])

    mix2_math_node = add_node(node_tree, 'ShaderNodeVectorMath', (0, -1.0))
    mix2_math_node.operation = 'MULTIPLY'
    node_tree.links.new(am2_mul_math_node.outputs[0], mix2_math_node.inputs[0])
    node_tree.links.new(sub2_math_node.outputs['Value'], mix2_math_node.inputs[1])
    node_tree.links.new(mix2_math_node.outputs[0], mix1_math_node.inputs[2])


def load_pbs_ext_material(node_tree: bpy.types.NodeTree, diff_node: bpy.types.ShaderNodeBsdfDiffuse, props: dict[str, Any], ctx: LoaderContext):
    diffuseMap = props.get('diffuseMap')
    if diffuseMap is None:
        return

    diffuse_tex_node = add_node(node_tree, 'ShaderNodeTexImage', (-2, 0))
    diffuse_tex_node.image = get_image(ctx, diffuseMap)
    node_tree.links.new(diffuse_tex_node.outputs['Color'], diff_node.inputs['Color'])


def load_pbs_tiled_material(node_tree: bpy.types.NodeTree, diff_node: bpy.types.ShaderNodeBsdfDiffuse, props: dict[str, Any], ctx: LoaderContext):
    albedoHeightTile0 = props.get('albedoHeightTile0')
    if albedoHeightTile0 is None:
        return

    albedoHeightTile1 = props.get('albedoHeightTile1')
    if albedoHeightTile1 is None:
        return

    albedoHeightTile2 = props.get('albedoHeightTile2')
    if albedoHeightTile2 is None:
        return

    g_atlasSizes = props.get('g_atlasSizes')
    if g_atlasSizes is None:
        return

    blendMask = props.get('blendMask')
    if blendMask is None:
        return

    # blendMask is always png
    if not blendMask.endswith('.png'):
        return

    g_tile0Tint = Vector(props.get('g_tile0Tint', (1, 1, 1, 1))[:3])
    g_tile1Tint = Vector(props.get('g_tile1Tint', (1, 1, 1, 1))[:3])
    g_tile2Tint = Vector(props.get('g_tile2Tint', (1, 1, 1, 1))[:3])

    uv2_node = add_node(node_tree, 'ShaderNodeUVMap', (-9, -1))
    uv2_node.uv_map = 'uv2'

    blendMask = blendMask[:-4] + '.dds'
    blendMask_img = get_image(ctx, blendMask)
    blendMask_img.colorspace_settings.name = 'Non-Color'
    blendMask_tex_node = add_node(node_tree, 'ShaderNodeTexImage', (-6, -1))
    blendMask_tex_node.image = blendMask_img
    node_tree.links.new(uv2_node.outputs['UV'], blendMask_tex_node.inputs['Vector'])

    sep_col_node = add_node(node_tree, 'ShaderNodeSeparateColor', (-3, -1))
    node_tree.links.new(blendMask_tex_node.outputs['Color'], sep_col_node.inputs['Color'])

    tile0_tex_node = add_node(node_tree, 'ShaderNodeTexImage', (-6, 0))
    tile0_tex_node.image = get_image(ctx, albedoHeightTile0)
    tile0_mul_node = add_node(node_tree, 'ShaderNodeVectorMath', (-3, 0))
    tile0_mul_node.operation = 'MULTIPLY'
    tile0_mul_node.inputs[1].default_value = g_tile0Tint
    node_tree.links.new(tile0_tex_node.outputs['Color'], tile0_mul_node.inputs[0])

    tile1_tex_node = add_node(node_tree, 'ShaderNodeTexImage', (-6, 1))
    tile1_tex_node.image = get_image(ctx, albedoHeightTile1)
    tile1_mul_node = add_node(node_tree, 'ShaderNodeVectorMath', (-3, 1))
    tile1_mul_node.operation = 'MULTIPLY'
    tile1_mul_node.inputs[1].default_value = g_tile1Tint
    node_tree.links.new(tile1_tex_node.outputs['Color'], tile1_mul_node.inputs[0])

    tile2_tex_node = add_node(node_tree, 'ShaderNodeTexImage', (-6, 2))
    tile2_tex_node.image = get_image(ctx, albedoHeightTile2)
    tile2_mul_node = add_node(node_tree, 'ShaderNodeVectorMath', (-3, 2))
    tile2_mul_node.operation = 'MULTIPLY'
    tile2_mul_node.inputs[1].default_value = g_tile2Tint
    node_tree.links.new(tile2_tex_node.outputs['Color'], tile2_mul_node.inputs[0])

    mul_math_node1 = add_node(node_tree, 'ShaderNodeVectorMath', (-1, 0))
    mul_math_node1.operation = 'MULTIPLY_ADD'

    mul_math_node2 = add_node(node_tree, 'ShaderNodeVectorMath', (-1, 1))
    mul_math_node2.operation = 'MULTIPLY_ADD'

    mul_math_node3 = add_node(node_tree, 'ShaderNodeVectorMath', (-1, 2))
    mul_math_node3.operation = 'MULTIPLY'

    node_tree.links.new(tile0_mul_node.outputs[0], mul_math_node1.inputs[0])
    node_tree.links.new(sep_col_node.outputs[0], mul_math_node1.inputs[1])
    node_tree.links.new(mul_math_node2.outputs[0], mul_math_node1.inputs[2])

    node_tree.links.new(tile1_mul_node.outputs[0], mul_math_node2.inputs[0])
    node_tree.links.new(sep_col_node.outputs[1], mul_math_node2.inputs[1])
    node_tree.links.new(mul_math_node3.outputs[0], mul_math_node2.inputs[2])

    node_tree.links.new(tile2_mul_node.outputs[0], mul_math_node3.inputs[0])
    node_tree.links.new(sep_col_node.outputs[2], mul_math_node3.inputs[1])

    node_tree.links.new(mul_math_node1.outputs[0], diff_node.inputs['Color'])


def load_material(name: str, umesh: UniversalMesh, ctx: LoaderContext):
    material = bpy.data.materials.new(name)
    material.use_nodes = True

    node_tree = material.node_tree
    node_tree.nodes.clear()

    out_node = node_tree.nodes.new('ShaderNodeOutputMaterial')
    out_node.location = (400.0, 0.0)

    diff_node = node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    diff_node.location = (200.0, 0.0)
    node_tree.links.new(diff_node.outputs['BSDF'], out_node.inputs['Surface'])

    # debug props
    material['fx_name'] = umesh.fx_name
    for key, value in umesh.props.items():
        material[key] = value

    match umesh.fx_name:
        case (
            'shaders/std_effects/PBS_ext.fx'
            | 'shaders/std_effects/PBS_ext_dual.fx'
            | 'shaders/std_effects/PBS_ext_repaint.fx'
            | 'shaders/std_effects/PBS_ext_detail.fx'
            | 'shaders/custom/coloronly_alpha.fx'
            | 'shaders/std_effects/PBS_ext_skinned_dual.fx'
            | 'shaders/std_effects/PBS_ext_skinned.fx'
            | 'shaders/std_effects/normalmap_specmap_dual.fx'
            | 'shaders/std_effects/normalmap_specmap.fx'
            | 'shaders/std_effects/normalmap_specmap_skinned.fx'
            | 'shaders/std_effects/normalmap.fx'
            | 'shaders/std_effects/lightonly.fx'
        ):
            load_pbs_ext_material(node_tree, diff_node, umesh.props, ctx)
        case 'shaders/std_effects/PBS_tiled.fx':
            load_pbs_tiled_material(node_tree, diff_node, umesh.props, ctx)
        case 'shaders/std_effects/PBS_tiled_atlas.fx' | 'shaders/std_effects/PBS_tiled_atlas_global.fx':
            load_pbs_tiled_atlas_material(node_tree, diff_node, umesh.props, ctx)
        case 'shaders/std_effects/PBS_glass.fx':
            # TODO...
            pass
        case _:
            logger.info(f'Unknown shader: {umesh.fx_name}')

    return material


def load_mesh(dataMesh: LoadDataMesh_v2, mesh: UniversalMesh, name: str) -> bpy.types.Mesh:
    verts = dataMesh.get_vertices_by_id(mesh.pg_idx)

    bmesh = bpy.data.meshes.new(f'mesh_{name}_{mesh.rset_id}_{mesh.pg_idx}')
    bmesh.vertices.add(len(verts))
    bmesh.vertices.foreach_set('co', verts.flatten())

    indices = dataMesh.get_indices_by_id(mesh.pg_idx)
    nbr_faces = len(indices)
    bmesh.polygons.add(nbr_faces)

    bmesh.polygons.foreach_set('loop_start', range(0, nbr_faces * 3, 3))
    bmesh.polygons.foreach_set('loop_total', (3,) * nbr_faces)

    bmesh.loops.add(nbr_faces * 3)
    bmesh.loops.foreach_set('vertex_index', indices.flatten())

    uv_layer = bmesh.uv_layers.new()
    uv_layer.name = 'uv1'
    uv_layer.active = True
    uv_layer = uv_layer.data[:]
    uv_list = dataMesh.get_uv_by_id(mesh.pg_idx)
    for poly in bmesh.polygons:
        for li in poly.loop_indices:
            vi = bmesh.loops[li].vertex_index
            uv_layer[li].uv = uv_list[vi]

    if dataMesh.uv2_list is not None:
        uv2_layer = bmesh.uv_layers.new()
        uv2_layer.name = 'uv2'
        uv2_layer = uv2_layer.data[:]
        uv2_list = dataMesh.get_uv2_by_id(mesh.pg_idx)
        for poly in bmesh.polygons:
            for li in poly.loop_indices:
                vi = bmesh.loops[li].vertex_index
                uv2_layer[li].uv = uv2_list[vi]

    return bmesh


def load_objects(col: bpy.types.Collection, ctx: LoaderContext):
    for model in ctx.space.models:
        if not ctx.res_mgr.exists(model.prims_name):
            logger.info(f'Cannot load: {model}')
            continue

        with ctx.res_mgr.open(model.prims_name) as f:
            dataMesh = LoadDataMesh_v2(f, model.verts_dataname, model.prims_dataname)

        name = bpy.path.display_name_from_filepath(model.prims_name)

        for instances in model.instances:
            for umesh in instances.meshes:
                bmesh = load_mesh(dataMesh, umesh, name)
                material = load_material(f'{name}_{umesh.pg_idx}', umesh, ctx)
                bmesh.materials.append(material)

                bmesh.transform(flip_mat)
                bmesh.validate()
                bmesh.update()

                for i, mat in enumerate(instances.transforms):
                    transposed_mat = Matrix((mat[0:4], mat[4:8], mat[8:12], mat[12:16])).transposed()
                    ob = bpy.data.objects.new(f'{name}_{umesh.rset_id}_{umesh.pg_idx}_{i}', bmesh)
                    # matrix magic
                    ob.matrix_world = flip_mat @ transposed_mat @ flip_mat
                    col.objects.link(ob)
