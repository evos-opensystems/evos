from pathlib import Path

from setuptools import setup, find_packages

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
)
