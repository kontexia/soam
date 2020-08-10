from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Distutils import build_ext
    import Cython.Compiler.Options
    from Cython.Build import cythonize

except ImportError:
    use_cython = False
else:
    use_cython = True
    Cython.Compiler.Options.annotate = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("src.neuro_column", ["src/neuro_column.py"]),
        Extension("src.neural_fabric", ["src/neural_fabric.py"]),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("src.neuro_column", ["src/neuro_column.c"]),
        Extension("src.neural_fabric", ["src/neural_fabric.c"]),
    ]

setup(
    name='soam',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    packages=['src', 'tests']
 )
