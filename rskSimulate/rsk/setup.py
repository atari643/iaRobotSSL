from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import rsk
extension = [
    Extension("game_controller", ["game_controller.py"])
]
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extension),
)