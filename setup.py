from pathlib import Path
import numpy as np
import os,subprocess
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

# extensions = [

#     Extension("*",["evos/ext/*.pyx"],
#         include_dirs = [np.get_include()], # generally not needed but typical use case 
#         extra_compile_args=['-O3','-march=native','-fopenmp','-Wno-cpp'], # generally not needed but typical use case
#         extra_link_args=['-fopenmp', '-lgomp', '-Wl,-rpath,'+str(subprocess.check_output([os.getenv('CC'), "-print-libgcc-file-name"]).strip().decode()).rstrip('/')] # might fix rare openmp bug
#         ),
# ]

setup(
	name = "evos",
	python_version = ">=3.6",
	description = "change me",
	long_description = open("README.md").read(),
	long_description_content_type = "text/markdown",
	author = "Réka Schwengelbeck, Mattia Moroder and Martin Grundner",
	author_email = "mattia.moroder@physik.uni-muenchen.de",
	license = 'BSD 2-clause',
	packages = find_packages(),
	install_requires = ["pytest>=0","pdoc>=0","numpy>=0","pybind11>=0"],
        #ext_modules=[system_cpp_module],
	classifiers = [
		"Developmet Status :: 1 - Planning",
		"Intended Audience :: Science/Research",
		"Operating System :: POSIX :: Linux",
		"Operating System :: MacOS :: MacOS X",
		"Programming Language :: Python :: 3",
	],
 	# ext_modules = cythonize(extensions, compiler_directives={'language_level' : 3}, annotate = True),
)
