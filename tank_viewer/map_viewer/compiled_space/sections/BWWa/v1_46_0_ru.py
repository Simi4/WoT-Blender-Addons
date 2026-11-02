""" BWWa (Water) """

from .v1_0_0 import BWWa_Section_1_0_0



class BWWa_Section_1_46_0_RU(BWWa_Section_1_0_0):
    int1 = 3

    _fields_ = [
        (list, '1', '<117I'),          # 468 bytes
        (list, '2', '<6f'),
        (list, '3', '<3f2If'),
        (list, '4', '<I'),
        (list, '5', '<b'),
    ]
