""" BWSV (visibilityMask) """
# EN: This section is deprecated and kept for backward compatibility.
# RU: Данная секция устарела. Она оставлена для обратной совместимости. 

from .._base_json_section import *



__all__ = ('BWSV_Section_0_9_12',)



class BWSV_Section_0_9_12(Base_JSON_Section):
	header = 'BWSV'
	int1 = 1

	_fields_ = [
		(list, 'visibility_masks', '<I' ),
		]
