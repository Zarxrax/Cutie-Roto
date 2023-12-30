#!python
#cython: language_level=3
import platform
if platform.system() != 'Windows':
    import pyximport
    pyximport.install(pyximport=True, language_level=3)

# noinspection PyUnresolvedReferences
from ._get_dist_maps import get_dist_maps