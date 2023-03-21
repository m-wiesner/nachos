from pathlib import Path
import importlib

__SPLITTERS_DICT__ = dict()

def register(name):
    def register_fn(cls):
        if name in __SPLITTERS_DICT__:
            raise(f"{name} is already registered")
        __SPLITTERS_DICT__[name] = cls
        return cls
    return register_fn


# Grab the lowercase similarity functions; they are not classes and just
# implement the functions 
for f in Path(__file__).parent.glob('[a-z_]*.py'):
    module_name = f.stem
    if module_name == "__init__":
        continue;
    module = importlib.import_module('nachos.splitters.' + module_name)


def build_splitter(conf):
    return __SPLITTERS_DICT__[conf['splitter']].build(conf)
