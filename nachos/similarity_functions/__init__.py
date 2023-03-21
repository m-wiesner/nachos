from pathlib import Path
import importlib


__SIMILARITY_FUNCTION_DICT__ = dict()


def register(name):
    def register_fn(cls):
        if name in __SIMILARITY_FUNCTION_DICT__:
            raise(f"{name} is already registered")
        __SIMILARITY_FUNCTION_DICT__[name] = cls
        return cls
    return register_fn


# Grab the lowercase similarity functions; they are not classes and just
# implement the functions 
for f in Path(__file__).parent.glob('[a-z_]*.py'):
    module_name = f.stem
    if module_name == "__init__":
        continue;
    module = importlib.import_module('nachos.similarity_functions.' + module_name)


from nachos.similarity_functions.SimilarityFunctions import SimilarityFunctions


def build_similarity_functions(conf):
    new_conf = dict(conf)
    sim_funs = conf['similarity_functions']
    # Update the conf to have the callable classes instead of just string names
    # of methods from the config file
    new_conf['similarity_functions'] = [
        __SIMILARITY_FUNCTION_DICT__[f] for f in sim_funs
    ]
    return SimilarityFunctions.build(new_conf) 
