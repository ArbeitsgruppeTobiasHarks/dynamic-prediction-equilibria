from setuptools import setup
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize("**/*.pyx", compiler_directives={"language_level": "3"}, annotate=True),
    zip_safe=False,
)
