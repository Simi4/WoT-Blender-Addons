
# imports
from ctypes import Structure, c_short, c_uint32
from dataclasses import dataclass
from functools import cache
from struct import calcsize, unpack
from typing import IO
from zipfile import ZipFile

# blender imports
import numpy as np
import bpy  # type: ignore
from mathutils import Matrix, Vector  # type: ignore

# local imports
from ..common import addon_prefs, load_image_from_memory, logger
from .loader_context import LoaderContext


@dataclass
class Layer:
    name: str
    name_nm: str | None
    uProjection: Vector
    vProjection: Vector
    row0: Vector
    row1: Vector
    row2: Vector


class DDS_HEADER(Structure):
    # fmt: off
    _fields_ = [
        ('dwSize',              c_uint32      ),
        ('dwFlags',             c_uint32      ),
        ('dwHeight',            c_uint32      ),
        ('dwWidth',             c_uint32      ),
        ('dwPitchOrLinearSize', c_uint32      ),
        ('dwDepth',             c_uint32      ),
        ('dwMipMapCount',       c_uint32      ),
        ('dwReserved1',         c_uint32 * 11 ),
        ('pf_Size',             c_uint32      ),
        ('pf_Flags',            c_uint32      ),
        ('pf_FourCC',           c_uint32      ),
        ('pf_RGBBitCount',      c_uint32      ),
        ('pf_RBitMask',         c_uint32      ),
        ('pf_GBitMask',         c_uint32      ),
        ('pf_BBitMask',         c_uint32      ),
        ('pf_ABitMask',         c_uint32      ),
        ('dwCaps',              c_uint32      ),
        ('dwCaps2',             c_uint32      ),
        ('dwCaps3',             c_uint32      ),
        ('dwCaps4',             c_uint32      ),
        ('dwReserved2',         c_uint32      ),
    ]
    # fmt: on


def create_mix_nodes(node_tree, count: int, y: float) -> list:
    nodes = []

    prev_node = None
    for i in range(count):
        math_node = node_tree.nodes.new('ShaderNodeVectorMath')
        if i != count - 1:
            math_node.operation = 'MULTIPLY_ADD'
        else:
            math_node.operation = 'MULTIPLY'
        math_node.location = (-300.0, y - i * 50.0)
        math_node.hide = True
        if i > 0:
            node_tree.links.new(math_node.outputs['Vector'], prev_node.inputs[2])
        nodes.append(math_node)
        prev_node = math_node

    return nodes


def create_nm_mix_nodes(node_tree, count: int, y: float) -> list:
    nodes = []

    prev_node = None
    for i in range(count):
        sep_col_node = node_tree.nodes.new('ShaderNodeSeparateColor')
        sep_col_node.location = (-300.0, y - i * 50.0)
        sep_col_node.hide = True

        comb_col_node = node_tree.nodes.new('ShaderNodeCombineColor')
        comb_col_node.location = (-100.0, y - i * 50.0)
        comb_col_node.hide = True

        math_node = node_tree.nodes.new('ShaderNodeVectorMath')
        if i != count - 1:
            math_node.operation = 'MULTIPLY_ADD'
        else:
            math_node.operation = 'MULTIPLY'
        math_node.location = (100.0, y - i * 50.0)
        math_node.hide = True
        if i > 0:
            node_tree.links.new(math_node.outputs['Vector'], prev_node.inputs[2])
        node_tree.links.new(sep_col_node.outputs['Red'], comb_col_node.inputs['Red'])
        node_tree.links.new(sep_col_node.outputs['Green'], comb_col_node.inputs['Green'])
        node_tree.links.new(comb_col_node.outputs['Color'], math_node.inputs[0])
        nodes.append((sep_col_node.inputs['Color'], comb_col_node.inputs['Blue'], math_node))
        prev_node = math_node

    return nodes


def get_layer_mapping_group():
    if 'TerrainLayerMappingGroup' in bpy.data.node_groups:
        return bpy.data.node_groups['TerrainLayerMappingGroup']

    group = bpy.data.node_groups.new('TerrainLayerMappingGroup', 'ShaderNodeTree')
    group.interface.new_socket(name='Vector', in_out='INPUT', socket_type='NodeSocketVector')
    group.interface.new_socket(name='Rotation', in_out='INPUT', socket_type='NodeSocketVector')
    group.interface.new_socket(name='Scale', in_out='INPUT', socket_type='NodeSocketVector')
    group.interface.new_socket(name='Vector', in_out='OUTPUT', socket_type='NodeSocketVector')

    group_in = group.nodes.new('NodeGroupInput')
    group_in.location = (-400.0, 0.0)

    tex_min = 0.0625
    tex_max = 0.9375

    main_mapping_node = group.nodes.new('ShaderNodeMapping')
    main_mapping_node.location = (-200.0, 0.0)
    main_mapping_node.vector_type = 'TEXTURE'
    main_mapping_node.inputs[1].default_value = Vector((tex_min, tex_min, 0.0))

    wrap_node = group.nodes.new('ShaderNodeVectorMath')
    wrap_node.operation = 'WRAP'
    wrap_node.location = (200.0, 0.0)

    wrap_node.inputs[1].default_value = Vector((tex_max, tex_max, 0.0))
    wrap_node.inputs[2].default_value = Vector((tex_min, tex_min, 0.0))

    group_out = group.nodes.new('NodeGroupOutput')
    group_out.location = (400.0, 0.0)

    group.links.new(group_in.outputs['Vector'], main_mapping_node.inputs['Vector'])
    group.links.new(group_in.outputs['Rotation'], main_mapping_node.inputs['Rotation'])
    group.links.new(group_in.outputs['Scale'], main_mapping_node.inputs['Scale'])
    group.links.new(main_mapping_node.outputs['Vector'], wrap_node.inputs['Vector'])
    group.links.new(wrap_node.outputs['Vector'], group_out.inputs['Vector'])
    return group


def unpack_blend_textures(fr, chunk_name: str) -> list:
    blend_textures = []
    header = fr.read(4)
    assert header == b'bwb\x00', header
    logger.debug(f'{header=}')

    section_count = unpack('<I', fr.read(4))[0]
    section_sizes = unpack('<4I', fr.read(4 * 4))
    logger.debug(f'{section_sizes=}')

    for i in range(section_count):
        header = fr.read(4)
        assert header == b'bwt\x00', header
        logger.debug(f'{header=}')

        version, xsize, ysize, always19, tex_cnt, padding = unpack('<IHHHHQ', fr.read(calcsize('<IHHHHQ')))
        logger.debug(f'{version=}')
        logger.debug(f'{xsize=}')
        logger.debug(f'{ysize=}')
        logger.debug(f'{tex_cnt=}')

        assert version == 2
        assert always19 == 19
        assert padding == 0

        for j in range(tex_cnt):
            name_size = unpack('<I', fr.read(4))[0]
            name = fr.read(name_size)
            logger.debug(f'{name=}')

        new_header = DDS_HEADER()
        new_header.dwSize = 124
        new_header.dwFlags = 0x1 | 0x2 | 0x4 | 0x1000
        new_header.dwHeight = ysize
        new_header.dwWidth = xsize
        new_header.dwMipMapCount = 0
        new_header.pf_Size = 32
        new_header.pf_Flags = 4  # FourCCFlag
        new_header.pf_FourCC = int.from_bytes(b'DXT5', 'little')
        new_header.dwCaps = 0x1000  # DDSCAPS_TEXTURE

        blend_tex_data = b'DDS ' + bytes(new_header) + fr.read(xsize * ysize)
        blend_tex = load_image_from_memory(blend_tex_data, f'{chunk_name}_blend_texture_{i}.dds')
        blend_tex.colorspace_settings.name = 'Non-Color'
        blend_textures.append(blend_tex)
    return blend_textures


@cache
def get_chunk_faces(width: int) -> list:
    faces = []
    for i in range(0, width * width - width - 1):
        if (i + 1) % width == 0:
            continue
        faces.append((i + 1, i, i + width, i + width + 1))
    return faces


def unpack_heights(fr, chunk_name: str, chunk_size: float):
    _ = fr.read(4)
    png_width, png_height = unpack('<2I', fr.read(8))
    fr.seek(36)

    raw = load_image_from_memory(fr.read(), f'{chunk_name}_heights1.png')

    rp = np.empty(png_width * png_height << 2, np.float32)
    raw.pixels.foreach_get(rp)
    bpy.data.images.remove(raw)

    rp = rp.reshape((png_width * png_height, 4))

    scaleFactor = 1000.0 / 256.0

    first = rp[:, 0]
    second = rp[:, 1]
    third = rp[:, 2]

    # FIXME: NEED TO CHECK THIS
    third = np.where(third > 0.5, third - 1.0039216, third)
    out = (first + (second * 256) + third * 65536) / scaleFactor
    out = out.reshape((png_width, png_height))

    verts = []

    XY = np.linspace(0, chunk_size, png_width)

    for ix in range(png_width):
        for iy in range(png_height):
            verts.append(((XY[ix], XY[iy], out[png_height - iy - 1][ix])))

    faces = get_chunk_faces(png_width)

    bmesh = bpy.data.meshes.new(f'Mesh_{chunk_name}')
    bmesh.from_pydata(verts, [], faces)

    return bmesh


def unpack_normals(fr, chunk_name: str):
    magic = fr.read(4)
    assert magic == b'nrm\x00', magic

    header = unpack('<6H', fr.read(3 * 4))
    if header[0] == 1:
        # version 1
        normals_tex_name = f'{chunk_name}_normals.png'
        normals_tex_data = fr.read()

    else:
        # version 2
        assert header[0] == 2, header
        assert header[1] == 0, header
        assert header[4] == 19, header
        width, height = header[2], header[3]

        new_header = DDS_HEADER()
        new_header.dwSize = 124
        new_header.dwFlags = 0x1 | 0x2 | 0x4 | 0x1000
        new_header.dwHeight = width
        new_header.dwWidth = height
        new_header.dwMipMapCount = 0
        new_header.pf_Size = 32
        new_header.pf_Flags = 4  # FourCCFlag
        new_header.pf_FourCC = int.from_bytes(b'DXT5', 'little')
        new_header.dwCaps = 0x1000  # DDSCAPS_TEXTURE

        normals_tex_name = f'{chunk_name}_normals.dds'
        normals_tex_data = b'DDS ' + bytes(new_header) + fr.read(width * height)

    normals_img = load_image_from_memory(normals_tex_data, normals_tex_name)
    normals_img.colorspace_settings.name = 'Non-Color'
    return normals_img


def unpack_layer(fr: IO[bytes], is_new_format: bool, chunk_name: str, i: int):
    header = fr.read(4)
    assert header == b'bld\x00', header
    logger.debug(f'{header=}')

    width, height, count = unpack('<3I', fr.read(3 * 4))
    logger.debug(f'{width=}')
    logger.debug(f'{height=}')
    logger.debug(f'{count=}')  # bpp?

    uProjection = Vector(unpack('<4f', fr.read(4 * 4)))
    vProjection = Vector(unpack('<4f', fr.read(4 * 4)))

    flags = unpack('<I', fr.read(4))[0]
    if is_new_format:
        assert flags == 0x3B, flags
    else:
        assert flags == 0x02, flags

    padding = unpack('<3I', fr.read(3 * 4))
    assert padding in ((0, 0, 0), (1, 0, 0)), padding

    # Displacement
    if is_new_format:
        row0 = Vector(unpack('<4f', fr.read(4 * 4)))
        row1 = Vector(unpack('<4f', fr.read(4 * 4)))
        row2 = Vector(unpack('<4f', fr.read(4 * 4)))
    else:
        # dirty hack!!!
        has_nm = padding[0] == 1
        row0 = None
        row1 = None
        row2 = None

    sz = unpack('<I', fr.read(4))[0]
    name = fr.read(sz).decode('utf-8').lower()

    if is_new_format:
        _ = fr.read(1)
        name_nm = name.replace('_am.', '_nm.')
        blend_tex = None
    else:
        if name.endswith('.tga'):
            name = name[:-3] + 'dds'
        if has_nm:
            sz = unpack('<I', fr.read(4))[0]
            name_nm = fr.read(sz).decode('utf-8')
        else:
            name_nm = None

        blend_tex = load_image_from_memory(fr.read(), f'{chunk_name}_blend_{i}.png')
        blend_tex.colorspace_settings.name = 'Non-Color'

    layer = Layer(name, name_nm, uProjection, vProjection, row0, row1, row2)
    return layer, blend_tex


def unpack_layers(fr: IO[bytes], chunk_name: str):
    layers = []
    header = fr.read(4)
    assert header == b'blb\x00', header
    logger.debug(f'{header=}')

    map_count = unpack('<I', fr.read(4))[0]
    section_sizes = unpack('<8I', fr.read(8 * 4))
    logger.debug(f'{map_count=}')
    logger.debug(f'{section_sizes=}')

    for i in range(map_count):
        layers.append(unpack_layer(fr, True, chunk_name, i)[0])

    return layers


def get_image(ctx: LoaderContext, name: str):
    name = name.lower()

    if _img := ctx.image_cache.get(name):
        return _img

    if item := ctx.res_mgr.pkgs.get(name):
        data = item[0].read(item[1])
        _img = load_image_from_memory(data, name)
        ctx.image_cache[name] = _img
        return _img

    logger.error(f'{name} not found!')
    return None


def load_terrain(col, ctx: LoaderContext):
    # center = map_info['boundingBox']['upperRight'] + map_info['boundingBox']['bottomLeft']
    geometry = ctx.map_info['geometry']

    bounds = ctx.space.terrain.bounds
    num_chunks = Vector(ctx.space.terrain.num_chunks)
    chunk_size = ctx.space.terrain.chunk_size

    global_AM_img = None
    if ctx.space.terrain.global_map:
        global_AM_img = get_image(ctx, ctx.space.terrain.global_map)
        global_AM_img.colorspace_settings.name = 'Non-Color'

    map_viewer_load_normals = addon_prefs().map_viewer_load_normals and global_AM_img
    map_viewer_load_wetness = addon_prefs().map_viewer_load_wetness and global_AM_img

    for item in (ctx.res_mgr.unp_dir / geometry).glob('*.cdata*'):
        chunk_name = item.name[:8]

        hexX, hexY = int(chunk_name[:4], 16), int(chunk_name[4:8], 16)
        hexX = c_short(hexX).value
        hexY = c_short(hexY).value
        chunk_pos = Vector((hexX, hexY)) * chunk_size

        with ZipFile(item, 'r') as zfile:
            if blend_textures_zinfo := zfile.NameToInfo.get('terrain2/blend_textures'):
                # new version
                # dds, with alpha
                new_blend_format = True
                with zfile.open(blend_textures_zinfo, 'r') as fr:
                    blend_textures = unpack_blend_textures(fr, chunk_name)

                with zfile.open('terrain2/layers', 'r') as fr:
                    layers = unpack_layers(fr, chunk_name)
            else:
                # old version
                # blend_textures stores inside 'layer 1', ...
                # png, without alpha
                new_blend_format = False
                layers = []
                blend_textures = []
                for i in range(1, 5):
                    if layer_zinfo := zfile.NameToInfo.get(f'terrain2/layer {i}'):
                        with zfile.open(layer_zinfo, 'r') as fr:
                            layer, blend_tex = unpack_layer(fr, False, chunk_name, i)
                            layers.append(layer)
                            blend_textures.append(blend_tex)
                    else:
                        break

            if not layers or not blend_textures:
                continue

            with zfile.open('terrain2/heights1', 'r') as fr:
                chunk_mesh = unpack_heights(fr, chunk_name, chunk_size)

            if map_viewer_load_normals:
                with zfile.open('terrain2/normals', 'r') as fr:
                    chunk_normals_img = unpack_normals(fr, chunk_name)

        material = bpy.data.materials.new(f'Material_{chunk_name}')
        material.use_nodes = True
        node_tree = material.node_tree
        node_tree.nodes.clear()

        blend_tex_frame = node_tree.nodes.new('NodeFrame')
        blend_tex_frame.label = 'Blend Textures'

        tile_tex_frame = node_tree.nodes.new('NodeFrame')
        tile_tex_frame.label = 'Tile Textures'

        out_node = node_tree.nodes.new('ShaderNodeOutputMaterial')
        out_node.location = (800.0, 0.0)

        diff_node = node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        diff_node.location = (400.0, 0.0)
        node_tree.links.new(diff_node.outputs['BSDF'], out_node.inputs['Surface'])

        am_mix_nodes = create_mix_nodes(node_tree, len(layers), 0)
        node_tree.links.new(am_mix_nodes[0].outputs['Vector'], diff_node.inputs['Base Color'])

        tex_coord_node = node_tree.nodes.new('ShaderNodeTexCoord')
        tex_coord_node.location = (-1800.0, 0.0)
        tex_coord_node.hide = True

        mapping_node = node_tree.nodes.new('ShaderNodeMapping')
        mapping_node.vector_type = 'TEXTURE'
        mapping_node.location = (-1600.0, 0.0)
        mapping_node.hide = True
        mapping_node.inputs['Scale'].default_value = Vector((1.0, -1.0, 1.0))
        node_tree.links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])

        if map_viewer_load_normals:
            nm_mix_nodes = create_nm_mix_nodes(node_tree, len(layers), -600)
            chunk_nm_tex_node = node_tree.nodes.new('ShaderNodeTexImage')
            chunk_nm_tex_node.image = chunk_normals_img
            chunk_nm_tex_node.location = (-1200.0, 400.0)
            chunk_nm_tex_node.hide = True
            node_tree.links.new(mapping_node.outputs['Vector'], chunk_nm_tex_node.inputs['Vector'])

            sep_col2_node = node_tree.nodes.new('ShaderNodeSeparateColor')
            sep_col2_node.location = (100.0, -550.0)
            sep_col2_node.hide = True
            node_tree.links.new(nm_mix_nodes[0][2].outputs['Vector'], sep_col2_node.inputs['Color'])
            node_tree.links.new(sep_col2_node.outputs['Red'], diff_node.inputs['Specular Tint'])

            comb_col_node = node_tree.nodes.new('ShaderNodeCombineColor')
            comb_col_node.location = (100.0, -500.0)
            comb_col_node.hide = True
            comb_col_node.inputs['Blue'].default_value = 1.0
            node_tree.links.new(sep_col2_node.outputs['Green'], comb_col_node.inputs['Red'])
            node_tree.links.new(sep_col2_node.outputs['Blue'], comb_col_node.inputs['Green'])

            math_norm_node = node_tree.nodes.new('ShaderNodeVectorMath')
            math_norm_node.operation = 'NORMALIZE'
            math_norm_node.location = (100.0, -450.0)
            math_norm_node.hide = True
            node_tree.links.new(comb_col_node.outputs['Color'], math_norm_node.inputs['Vector'])

            normal_map_node = node_tree.nodes.new('ShaderNodeNormalMap')
            normal_map_node.location = (100.0, -400.0)
            normal_map_node.space = 'OBJECT'
            normal_map_node.hide = True
            normal_map_node.inputs[0].default_value = 0.1
            node_tree.links.new(math_norm_node.outputs['Vector'], normal_map_node.inputs['Color'])
            node_tree.links.new(normal_map_node.outputs['Normal'], diff_node.inputs['Normal'])

        chunk_uv_math_node = node_tree.nodes.new('ShaderNodeVectorMath')
        chunk_uv_math_node.operation = 'MULTIPLY_ADD'
        chunk_uv_math_node.location = (-1600.0, 600.0)
        chunk_uv_math_node.hide = True
        chunk_uv_math_node.inputs[1].default_value = Vector((1 / num_chunks.x, 1 / num_chunks.y, 0.0))
        chunk_uv_math_node.inputs[2].default_value = Vector(((hexX - bounds[0]) / num_chunks.x, (hexY - bounds[2]) / num_chunks.y, 0.0))
        node_tree.links.new(tex_coord_node.outputs['Generated'], chunk_uv_math_node.inputs['Vector'])

        if map_viewer_load_wetness:
            chunk_global_am_tex_node = node_tree.nodes.new('ShaderNodeTexImage')
            chunk_global_am_tex_node.image = global_AM_img
            chunk_global_am_tex_node.extension = 'CLIP'
            # FIXME(?): chunk_global_am_tex_node.interpolation = 'CUBIC'
            chunk_global_am_tex_node.location = (-1200.0, 600.0)
            chunk_global_am_tex_node.hide = True
            node_tree.links.new(chunk_uv_math_node.outputs['Vector'], chunk_global_am_tex_node.inputs['Vector'])

            global_am_sub_math_node = node_tree.nodes.new('ShaderNodeMath')
            global_am_sub_math_node.operation = 'SUBTRACT'
            global_am_sub_math_node.location = (-800.0, 600.0)
            global_am_sub_math_node.hide = True
            global_am_sub_math_node.inputs[0].default_value = 1.0
            node_tree.links.new(chunk_global_am_tex_node.outputs['Alpha'], global_am_sub_math_node.inputs[1])
            node_tree.links.new(global_am_sub_math_node.outputs['Value'], diff_node.inputs['Roughness'])

            global_am_mul_math_node = node_tree.nodes.new('ShaderNodeVectorMath')
            global_am_mul_math_node.operation = 'MULTIPLY'
            global_am_mul_math_node.location = (0.0, 0.0)
            global_am_mul_math_node.hide = True
            node_tree.links.new(chunk_global_am_tex_node.outputs['Color'], global_am_mul_math_node.inputs[0])
            node_tree.links.new(am_mix_nodes[0].outputs['Vector'], global_am_mul_math_node.inputs[1])
            node_tree.links.new(global_am_mul_math_node.outputs[0], diff_node.inputs['Base Color'])
        else:
            diff_node.inputs['Roughness'].default_value = 1.0

        world_uv_node = node_tree.nodes.new('ShaderNodeVectorMath')
        world_uv_node.operation = 'MULTIPLY_ADD'
        world_uv_node.location = (-1600.0, -200.0)
        world_uv_node.hide = True
        world_uv_node.inputs[1].default_value = Vector((chunk_size, chunk_size, 0.0))
        world_uv_node.inputs[2].default_value = Vector((chunk_pos.x, chunk_pos.y, 0.0))
        node_tree.links.new(tex_coord_node.outputs['Generated'], world_uv_node.inputs['Vector'])

        for i, layer in enumerate(layers):
            layer_am_img = get_image(ctx, layer.name)
            layer_am_img.alpha_mode = 'PREMUL'
            if map_viewer_load_normals:
                layer_nm_img = get_image(ctx, layer.name_nm)
                if layer_nm_img:
                    layer_nm_img.colorspace_settings.name = 'Non-Color'

            c1 = layer.uProjection.xyz.cross(layer.vProjection.xyz).normalized()
            m = Matrix()
            m.col[0] = layer.uProjection
            m.col[1] = Vector((c1.x, c1.y, c1.z, 0.0))
            m.col[2] = layer.vProjection
            m.invert()

            # dirty hack for old terrain version!!!
            if not new_blend_format and 'color_tex' in layer.name:
                mapping_layer_node = chunk_uv_math_node
            else:
                mapping_layer_node = node_tree.nodes.new('ShaderNodeGroup')
                mapping_layer_node.node_tree = get_layer_mapping_group()
                mapping_layer_node.inputs[1].default_value = Vector((0, 0, -m.to_euler().y))
                mapping_layer_node.inputs[2].default_value = Vector((-m.to_scale().x, -m.to_scale().z, 0))
                mapping_layer_node.location = (-1200.0, -400.0 - i * 50)
                mapping_layer_node.hide = True
                node_tree.links.new(world_uv_node.outputs['Vector'], mapping_layer_node.inputs['Vector'])

            am_tex_node = node_tree.nodes.new('ShaderNodeTexImage')
            am_tex_node.image = layer_am_img
            am_tex_node.parent = tile_tex_frame
            am_tex_node.location = (-800.0, -400.0 - i * 100)
            am_tex_node.hide = True

            if map_viewer_load_normals:
                nm_tex_node = node_tree.nodes.new('ShaderNodeTexImage')
                nm_tex_node.image = layer_nm_img
                nm_tex_node.parent = tile_tex_frame
                nm_tex_node.location = (-800.0, -450.0 - i * 100)
                nm_tex_node.hide = True
                node_tree.links.new(mapping_layer_node.outputs['Vector'], nm_tex_node.inputs['Vector'])

                node_tree.links.new(nm_tex_node.outputs['Alpha'], nm_mix_nodes[i][1])
                node_tree.links.new(nm_tex_node.outputs['Color'], nm_mix_nodes[i][0])

            node_tree.links.new(mapping_layer_node.outputs['Vector'], am_tex_node.inputs['Vector'])
            node_tree.links.new(am_tex_node.outputs['Color'], am_mix_nodes[i].inputs[0])

        for i, blend_tex in enumerate(blend_textures):
            tex_node = node_tree.nodes.new('ShaderNodeTexImage')
            tex_node.image = blend_tex
            tex_node.parent = blend_tex_frame
            tex_node.location = (-1200.0, 200.0 - i * 50)
            tex_node.hide = True
            node_tree.links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
            if new_blend_format:
                sep_col_node = node_tree.nodes.new('ShaderNodeSeparateColor')
                sep_col_node.location = (-800.0, 200.0 - i * 50)
                sep_col_node.hide = True
                node_tree.links.new(tex_node.outputs['Color'], sep_col_node.inputs['Color'])
                node_tree.links.new(tex_node.outputs['Alpha'], am_mix_nodes[i * 2].inputs[1])
                if map_viewer_load_normals:
                    node_tree.links.new(tex_node.outputs['Alpha'], nm_mix_nodes[i * 2][2].inputs[1])
                if i * 2 + 1 < len(am_mix_nodes):
                    node_tree.links.new(sep_col_node.outputs['Green'], am_mix_nodes[i * 2 + 1].inputs[1])
                    if map_viewer_load_normals:
                        node_tree.links.new(sep_col_node.outputs['Green'], nm_mix_nodes[i * 2 + 1][2].inputs[1])
            else:
                node_tree.links.new(tex_node.outputs['Color'], am_mix_nodes[i].inputs[1])

        chunk_mesh.validate()
        chunk_mesh.update()
        chunk_mesh.materials.append(material)

        ob = bpy.data.objects.new(chunk_name, chunk_mesh)
        ob.location.xy = chunk_pos
        col.objects.link(ob)
