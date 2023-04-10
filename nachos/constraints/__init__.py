from pathlib import Path
import importlib


__CONSTRAINTS_DICT__ = dict()


def register(name):
    def register_fn(cls):
        if name in __CONSTRAINTS_DICT__:
            raise(f"{name} is already registered")
        __CONSTRAINTS_DICT__[name] = cls
        return cls
    return register_fn


# Grab the lowercase similarity functions; they are not classes and just
# implement the functions 
for f in Path(__file__).parent.glob('[_a-z]*.py'):
    module_name = f.stem
    if module_name == "__init__":
        continue;
    module = importlib.import_module('nachos.constraints.' + module_name)


from nachos.constraints.Constraints import Constraints


def build_constraints(conf):
    new_conf = dict(conf)
    constraints = conf['constraints']
    new_conf['constraints'] = [
        {
            'name': __CONSTRAINTS_DICT__[c['name']],
            'values': c['values'] if 'values' in c else None,
            'reduction': c['reduction'],
        } for c in constraints
    ]
    return Constraints.build(new_conf) 
