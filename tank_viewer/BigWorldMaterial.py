


#####################################################################
# imports

from .common import *

import bpy


class BWNodeTree:

    def __init__(self, material):
        material.use_nodes = True

        self.next_tex_loc = [-1200.0, 200.0]
        self.next_const_loc = [-1900.0, 200.0]

        #maybe weakref?
        self.__material = material
        self.__node_tree = material.node_tree
        self.__node_tree.nodes.clear()

        # defaults
        tex_frame = self.n_new('NodeFrame')
        tex_frame.label = 'Textures'
        self.tex_frame = tex_frame

        consts_frame = self.n_new('NodeFrame')
        consts_frame.label = 'Constants'
        consts_frame.location = self.next_const_loc
        self.consts_frame = consts_frame

        uv1_node = self.n_new('ShaderNodeUVMap')
        uv1_node.uv_map = 'uv1'
        uv1_node.location = (-1600.0, -100.0)
        self.uv1_node = uv1_node

        out_node = self.n_new('ShaderNodeOutputMaterial')
        out_node.location = (400.0, 0.0)
        self.out_node = out_node

        princilpled_node = self.n_new('ShaderNodeBsdfPrincipled')
        princilpled_node.location = (-200.0, 200.0)
        self.l_new(princilpled_node.outputs['BSDF'], out_node.inputs['Surface'])
        self.princilpled_node = princilpled_node

    def n_new(self, *args):
        return self.__node_tree.nodes.new(*args)

    def l_new(self, *args):
        return self.__node_tree.links.new(*args)

    def addTexture(self, label, image, color_space = 'sRGB', uv_vec=None):
        tex_node = self.n_new('ShaderNodeTexImage')
        tex_node.image = image
        tex_node.location = self.next_tex_loc
        tex_node.image.colorspace_settings.name = color_space
        self.next_tex_loc[1] -= 300.0
        tex_node.parent = self.tex_frame
        tex_node.label = label
        if not uv_vec:
            uv_vec = self.uv1_node.outputs['UV']
        self.l_new(uv_vec, tex_node.inputs['Vector'])
        return tex_node

    def addFloatConst(self, label, value):
        value_node = self.n_new('ShaderNodeValue')
        value_node.outputs[0].default_value = value
        value_node.location = self.next_const_loc
        self.next_const_loc[1] -= 100.0
        value_node.parent = self.consts_frame
        value_node.label = label
        return value_node


#####################################################################
# setBigWorldMaterial

def setBigWorldMaterial(
        _shader_name, material,
        uv2_name, propertiesGroup):

    tree = BWNodeTree(material)

    checkTex = lambda name: propertiesGroup.get(name) and propertiesGroup[name].get('Texture') is not None
    checkFloat = lambda name: propertiesGroup.get(name) and propertiesGroup[name].get('Float') is not None
    checkBool = lambda name: propertiesGroup.get(name) and propertiesGroup[name].get('Bool') is not None

    alphaTestEnable = False
    if checkBool('alphaTestEnable'):
        alphaTestEnable = propertiesGroup['alphaTestEnable']['Bool']

    for name in ('g_maskBias', 'g_detailPower', 'g_detailPowerAlbedo', 'g_detailPowerGloss'):
        if checkFloat(name):
            tree.addFloatConst(name, propertiesGroup[name]['Float'])

    if _shader_name in (
            'PBS_tank_skinned.fx',
            'PBS_wheel_skinned.fx',
            'PBS_tank.fx',
            'PBS_tank_tracks.fx',
            'PBS_tank_uvtransform_skinned_ao.fx',
            'PBS_ext_dual.fx',
            'PBS_ext.fx',
            'PBS_ext_detail.fx',
            'PBS_tank_crash.fx',
            'PBS_tank_damage.fx',
            'PBS_tank_precise_edge.fx',
            'PBS_tank_skinned_ao.fx'):

        if checkTex('diffuseMap'):
            img_AM = propertiesGroup['diffuseMap']['Texture']
            tex_AM_node = tree.addTexture('diffuseMap', img_AM)
            tree.l_new(tex_AM_node.outputs['Color'], tree.princilpled_node.inputs['Base Color'])

            if checkTex('excludeMaskAndAOMap'):
                img_AO = propertiesGroup['excludeMaskAndAOMap']['Texture']
                img_AO.alpha_mode = 'NONE'
                tex_AO_node = tree.addTexture('excludeMaskAndAOMap', img_AO, 'Non-Color')

                mul_color_node = tree.n_new('ShaderNodeMixRGB')
                mul_color_node.blend_type = 'MULTIPLY'
                mul_color_node.inputs[0].default_value = 1.0
                mul_color_node.location = (-400.0, 200.0)

                sepRGB_node = tree.n_new('ShaderNodeSeparateColor')
                sepRGB_node.location = (-600.0, 0.0)

                tree.l_new(tex_AO_node.outputs['Color'], sepRGB_node.inputs['Color'])
                tree.l_new(tex_AM_node.outputs['Color'], mul_color_node.inputs['Color1'])
                tree.l_new(sepRGB_node.outputs['Green'], mul_color_node.inputs['Color2'])
                tree.l_new(mul_color_node.outputs['Color'], tree.princilpled_node.inputs['Base Color'])

            elif _shader_name.endswith('_skinned_ao.fx'):
                mul_color_node = tree.n_new('ShaderNodeMixRGB')
                mul_color_node.blend_type = 'MULTIPLY'
                mul_color_node.inputs[0].default_value = 1.0
                mul_color_node.location = (-400.0, 200.0)
                img_AM.alpha_mode = 'CHANNEL_PACKED'

                tree.l_new(tex_AM_node.outputs['Color'], mul_color_node.inputs['Color1'])
                tree.l_new(tex_AM_node.outputs['Alpha'], mul_color_node.inputs['Color2'])
                tree.l_new(mul_color_node.outputs['Color'], tree.princilpled_node.inputs['Base Color'])

        if _shader_name in ('PBS_ext_dual.fx',):
            if checkTex('diffuseMap2') and uv2_name:
                uv2_node = tree.n_new('ShaderNodeUVMap')
                uv2_node.uv_map = uv2_name
                uv2_node.location = (-1400.0, -1000.0)
                tree.addTexture('diffuseMap2', propertiesGroup['diffuseMap2']['Texture'], uv_vec=uv2_node.outputs['UV'])

        if checkTex('metallicGlossMap'):
            tex_GMM_node = tree.addTexture('metallicGlossMap', propertiesGroup['metallicGlossMap']['Texture'], 'Non-Color')

            sepRGB_node2 = tree.n_new('ShaderNodeSeparateColor')
            sepRGB_node2.location = (-800.0, -200.0)

            invert_node = tree.n_new('ShaderNodeMath')
            invert_node.location = (-600.0, -300.0)
            invert_node.operation = 'SUBTRACT'
            invert_node.inputs[0].default_value = 1.0

            tree.l_new(tex_GMM_node.outputs['Color'], sepRGB_node2.inputs['Color'])
            tree.l_new(sepRGB_node2.outputs['Green'], tree.princilpled_node.inputs['Metallic'])
            tree.l_new(sepRGB_node2.outputs['Red'], invert_node.inputs[1])
            tree.l_new(invert_node.outputs[0], tree.princilpled_node.inputs['Roughness'])

        if checkTex('normalMap'):
            if propertiesGroup.get('g_useNormalPackDXT1'):
                img_ANM = propertiesGroup['normalMap']['Texture']
                tex_ANM_Node = tree.addTexture('normalMap', img_ANM, 'Non-Color')

                normalMap_node = tree.n_new('ShaderNodeNormalMap')
                normalMap_node.location = (-400.0, -600.0)
                normalMap_node.uv_map = 'uv1'
                normalMap_node.inputs[0].default_value = 0.5
                tree.l_new(normalMap_node.outputs['Normal'], tree.princilpled_node.inputs['Normal'])

                displacement_Node = tree.n_new('ShaderNodeDisplacement')
                displacement_Node.location = (-200.0, -600.0)
                tree.l_new(normalMap_node.outputs['Normal'], displacement_Node.inputs['Normal'])
                tree.l_new(displacement_Node.outputs['Displacement'], tree.out_node.inputs['Displacement'])
                displacement_Node.inputs['Scale'].default_value = 0.05

                if propertiesGroup['g_useNormalPackDXT1']['Bool']:pass
                else:
                    sepRGB_node3 = tree.n_new('ShaderNodeSeparateColor')
                    sepRGB_node3.location = (-800.0, -700.0)

                    combineRGB_Node = tree.n_new('ShaderNodeCombineColor')
                    combineRGB_Node.location = (-600.0, -600.0)

                    tree.l_new(tex_ANM_Node.outputs['Alpha'], combineRGB_Node.inputs['Red'])
                    tree.l_new(sepRGB_node3.outputs['Green'], combineRGB_Node.inputs['Green'])
                    combineRGB_Node.inputs['Blue'].default_value = 1.0

                    tree.l_new(tex_ANM_Node.outputs['Color'], sepRGB_node3.inputs['Color'])
                    tree.l_new(combineRGB_Node.outputs['Color'], normalMap_node.inputs['Color'])

                    if alphaTestEnable:
                        alphaReference = 0.0
                        if 'alphaReference' in propertiesGroup:
                            alphaReference = propertiesGroup['alphaReference']['Int']/255.0

                        value_node = tree.addFloatConst('alphaReference', alphaReference)

                        lessthan_node = tree.n_new('ShaderNodeMath')
                        lessthan_node.location = (-400.0, -800.0)
                        lessthan_node.operation = 'GREATER_THAN'
                        tree.l_new(value_node.outputs[0], lessthan_node.inputs[1])

                        tree.l_new(sepRGB_node3.outputs['Red'], lessthan_node.inputs['Value'])
                        tree.l_new(lessthan_node.outputs['Value'], tree.princilpled_node.inputs['Alpha'])

        if checkBool('g_useDetailMetallic') and propertiesGroup['g_useDetailMetallic']['Bool']:
            g_detailUVTilingProp = propertiesGroup.get('g_detailUVTiling')

            if checkTex('metallicDetailMap') and g_detailUVTilingProp:
                Mapping_Node = tree.n_new('ShaderNodeMapping')
                Mapping_Node.vector_type = 'VECTOR'
                # TODO:
                #Mapping_Node.translation[:2] = g_detailUVTilingProp['Vector4'][2:4]
                #Mapping_Node.scale[:2] = g_detailUVTilingProp['Vector4'][:2]
                Mapping_Node.hide = True

                tree.l_new(tree.uv1_node.outputs['UV'], Mapping_Node.inputs['Vector'])

                tex_MDM_Node = tree.addTexture('metallicDetailMap', propertiesGroup['metallicDetailMap']['Texture'], uv_vec=Mapping_Node.outputs['Vector'])
                Mapping_Node.location = (-1400.0, tex_MDM_Node.location[1])

                # TODO

    elif _shader_name in (
            'PBS_tank_old_uvtransform_skinned.fx',
            'PBS_tank_old_skinned.fx',
            'PBS_tank_old.fx',
            'lightonly.fx'):

        if checkTex('diffuseMap'):
            tex_DM_Node = tree.n_new('ShaderNodeTexImage')
            tex_DM_Node.image = propertiesGroup['diffuseMap']['Texture']
            tex_DM_Node.location = (-400.0, 200.0)

            tree.l_new(tree.uv1_node.outputs['UV'], tex_DM_Node.inputs['Vector'])

            diffuseMap_Node = tree.n_new('ShaderNodeBsdfDiffuse')
            diffuseMap_Node.location = (-200.0, 200.0)
            diffuseMap_Node.inputs['Roughness'].default_value = 0.6

            tree.l_new(tex_DM_Node.outputs['Color'], diffuseMap_Node.inputs['Color'])

        if checkTex('specularMap'):
            tex_SM_Node = tree.n_new('ShaderNodeTexImage')
            tex_SM_Node.image = propertiesGroup['specularMap']['Texture']
            tex_SM_Node.location = (-600.0, -100.0)

            tree.l_new(tree.uv1_node.outputs['UV'], tex_SM_Node.inputs['Vector'])

            specularMap_Node = tree.n_new('ShaderNodeBsdfGlossy')
            specularMap_Node.location = (-200.0, -200.0)

            tree.l_new(tex_SM_Node.outputs['Color'], specularMap_Node.inputs['Roughness'])

            #tree.l_new(tex_SM_Node.outputs['Color'], node_mix_shader.inputs['Fac'])

            #tree.l_new(specularMap_Node.outputs['BSDF'], node_mix_shader.inputs[2])

        if checkTex('normalMap'):
            tex_NM_Node = tree.n_new('ShaderNodeTexImage')
            tex_NM_Node.image = propertiesGroup['normalMap']['Texture']
            tex_NM_Node.location = (-600.0, -400.0)

            tree.l_new(tree.uv1_node.outputs['UV'], tex_NM_Node.inputs['Vector'])

            normalMap_Node = tree.n_new('ShaderNodeNormalMap')
            normalMap_Node.location = (-200.0, -400.0)
            normalMap_Node.inputs[0].default_value = 0.25

            tree.l_new(tex_NM_Node.outputs['Color'], normalMap_Node.inputs['Color'])

            displacement_Node = tree.n_new('ShaderNodeDisplacement')
            tree.l_new(normalMap_Node.outputs['Normal'], displacement_Node.inputs['Normal'])
            tree.l_new(displacement_Node.outputs['Displacement'], tree.out_node.inputs['Displacement'])

    else:
        logger.warning(f'shader_name: {_shader_name}')
