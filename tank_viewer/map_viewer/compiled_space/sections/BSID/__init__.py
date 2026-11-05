""" BSID (BigWorld Static Index Data) """

from .._base_binary_section import *


__all__ = ('BSID_Section_1_46_0_RU',)



class BSID_Section_1_46_0_RU(Base_Binary_Section):
    header = 'BSID'
    int1 = 2
