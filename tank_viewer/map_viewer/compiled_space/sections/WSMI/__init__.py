""" WSMI (WoT Static Model Instances) """
# EN: This section is deprecated and kept for backward compatibility.
# RU: Данная секция устарела. Она оставлена для обратной совместимости. 

from .._base_json_section import *



__all__ = ('WSMI_Section_0_9_12',)



class WSMI_Section_0_9_12(Base_JSON_Section):
	header = 'WSMI'
	int1 = 1

	_fields_ = [
		(list, '1', '<3i' ),
		(list, '2', '<I'  ),
		]
