"""Observables class should depend only from Representation (ED OR MPS) class"""

import pkgutil

# Reference, How to import all submodules: https://stackoverflow.com/a/3365846/3211506

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module = loader.find_module(module_name).load_module(module_name)
    globals()[module_name] = _module