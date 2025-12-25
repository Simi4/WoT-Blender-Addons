

# original: https://github.com/konlil/mypy/blob/master/vertices.py


#####################################################################
# vt_baseType

class vt_baseType:
    uv2 = None

    FORMAT = ''
    SIZE = 0
    IS_SKINNED = False
    IS_NEW = False
    V_TYPE = ''
    DTYPE = None

    def __init__(self, t):
        self.__tuple = t

    def __str__(self):
        return str(self.__tuple)


#####################################################################
# vt_SET3_XYZNUVTB

class vt_SET3_XYZNUVTBPC(vt_baseType):
    FORMAT = '<3fI2f2I'
    SIZE = 32
    IS_SKINNED = False
    IS_NEW = True
    V_TYPE = 'set3/xyznuvtbpc'
    DTYPE=[('pos', '<f4', 3), ('packed_normal', '<i4'), ('uv', '<f4', 2), ('packed_tb', '<i4', 2)]

    def __init__(self, t):
        super(vt_SET3_XYZNUVTBPC, self).__init__(t)
        self.pos = (t[0], t[2], t[1])
        self.normal = t[3]
        self.uv = (t[4], 1-t[5])
        self.tangent = t[6]
        self.binormal = t[7]

    def to_tuple(self):
        if self.uv2 is not None:
            return (self.pos, self.normal, self.uv, self.uv2, self.tangent, self.binormal)
        return (self.pos, self.normal, self.uv, self.tangent, self.binormal)


#####################################################################
# vt_SET3_XYZNUVPC

class vt_SET3_XYZNUVPC(vt_baseType):
    FORMAT = '<3fI2f'
    SIZE = 24
    IS_SKINNED = False
    IS_NEW = True
    V_TYPE = 'set3/xyznuvpc'
    DTYPE=[('pos', '<f4', 3), ('packed_normal', '<i4'), ('uv', '<f4', 2)]

    def __init__(self, t):
        super(vt_SET3_XYZNUVPC, self).__init__(t)
        self.pos = (t[0], t[2], t[1])
        self.normal = t[3]
        self.uv = (t[4], 1-t[5])

    def to_tuple(self):
        if self.uv2 is not None:
            return (self.pos, self.normal, self.uv, self.uv2)
        return (self.pos, self.normal, self.uv)


#####################################################################
# vt_SET3_XYZNUVIIIWWTBPC

class vt_SET3_XYZNUVIIIWWTBPC(vt_baseType):
    FORMAT = '<3fI2f8B2I'
    SIZE = 40
    IS_SKINNED = True
    IS_NEW = True
    V_TYPE = 'set3/xyznuviiiwwtbpc'
    DTYPE=[('pos', '<f4', 3), ('packed_normal', '<u4'), ('uv', '<f4', 2), ('iiiww', 'B', 8), ('packed_tb', '<u4', 2)]

    def __init__(self, t):
        super(vt_SET3_XYZNUVIIIWWTBPC, self).__init__(t)
        self.pos = (t[0], -t[2], t[1])
        self.normal = t[3]
        self.uv = (t[4], 1-t[5])
        self.index = t[6]
        self.index2 = t[7]
        self.index3 = t[8]
        self.indexB_1 = t[9]
        self.indexB_2 = t[10]
        self.indexB_3 = t[11]
        self.weight = t[12]
        self.weight2 = t[13]
        self.tangent = t[14]
        self.binormal = t[15]

    def to_tuple(self):
        # if self.uv2 is not None:
        #     return (self.pos, self.normal, self.uv, self.uv2, self.index, self.index2, self.index3, self.indexB_1, self.indexB_2, self.indexB_3, self.weight, self.weight2, self.tangent, self.binormal)
        # return (self.pos, self.normal, self.uv, self.index, self.index2, self.index3, self.indexB_1, self.indexB_2, self.indexB_3, self.weight, self.weight2, self.tangent, self.binormal)
        if self.uv2 is not None:
            return (self.pos, self.normal, self.uv, self.uv2, self.tangent, self.binormal)
        return (self.pos, self.normal, self.uv, self.tangent, self.binormal)


#####################################################################
# vt_XYZNUVIIIWWTB

class vt_XYZNUVIIIWWTB(vt_baseType):
    FORMAT = '<3fI2f5B2I'
    SIZE = 37
    IS_SKINNED = True
    IS_NEW = False
    V_TYPE = 'xyznuviiiwwtb'
    DTYPE=[('pos', '<f4', 3), ('packed_normal', '<u4'), ('uv', '<f4', 2), ('iiiww', 'B', 5), ('packed_tb', '<u4', 2)]

    def __init__(self, t):
        super(vt_XYZNUVIIIWWTB, self).__init__(t)
        self.pos = (t[0], -t[2], t[1])
        self.normal = t[3]
        self.uv = (t[4], 1-t[5])
        self.index = t[6]
        self.index2 = t[7]
        self.index3 = t[8]
        self.weight = t[9]
        self.weight2 = t[10]
        self.tangent = t[11]
        self.binormal = t[12]

    def to_tuple(self):
        # if self.uv2 is not None:
        #     return (self.pos, self.normal, self.uv, self.uv2, self.index, self.index2, self.index3, self.weight, self.weight2, self.tangent, self.binormal)
        # return (self.pos, self.normal, self.uv, self.index, self.index2, self.index3, self.weight, self.weight2, self.tangent, self.binormal)
        if self.uv2 is not None:
            return (self.pos, self.normal, self.uv, self.uv2, self.tangent, self.binormal)
        return (self.pos, self.normal, self.uv, self.tangent, self.binormal)


#####################################################################
# vt_XYZNUVTB

class vt_XYZNUVTB(vt_baseType):
    FORMAT = '<3fI2f2I'
    SIZE = 32
    IS_SKINNED = False
    IS_NEW = False
    V_TYPE = 'xyznuvtb'
    DTYPE=[('pos', '<f4', 3), ('packed_normal', '<u4'), ('uv', '<f4', 2), ('packed_tb', '<u4', 2)]

    def __init__(self, t):
        super(vt_XYZNUVTB, self).__init__(t)
        self.pos = (t[0], t[2], t[1])
        self.normal = t[3]
        self.uv = (t[4], 1-t[5])
        self.tangent = t[6]
        self.binormal = t[7]

    def to_tuple(self):
        if self.uv2 is not None:
            return (self.pos, self.normal, self.uv, self.uv2, self.tangent, self.binormal)
        return (self.pos, self.normal, self.uv, self.tangent, self.binormal)


#####################################################################
# vt_XYZNUV

class vt_XYZNUV(vt_baseType):
    FORMAT = '<8f'
    SIZE = 32
    IS_SKINNED = False
    IS_NEW = False
    V_TYPE = 'xyznuv'
    DTYPE=[('pos', '<f4', 3), ('normal', '<f4', 3), ('uv', '<f4', 2)]

    def __init__(self, t):
        super(vt_XYZNUV, self).__init__(t)
        self.pos = (t[0], t[2], t[1])
        self.normal = (t[3], t[4], t[5])
        self.uv = (t[6], 1-t[7])

    def to_tuple(self):
        if self.uv2 is not None:
            return (self.pos, self.normal, self.uv, self.uv2)
        return (self.pos, self.normal, self.uv)
