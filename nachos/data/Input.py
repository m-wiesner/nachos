from nachos.data.Data import Data, Dataset


class TSVLoader(object):
    @staticmethod
    def load(fname, config):
        data = []
        with open(fname, 'r') as f:
            headers = f.readline().strip().split('\t')
            # Read each row
            for l in f:
                # 1st column is record, next columns are factors
                record, factors = l.strip().split('\t', 1)
                factors = factors.split('\t')
                # Each factor may be multivalued. We represent this as a set
                factors = [
                    set(
                        eval(config['factor_types'][i])(f)
                        for f in factors[i].split(',')
                    ) 
                    for i in range(len(factors))
                ]
                data.append(Data(record, factors, field_names=headers))
        return Dataset(data, config['factor_idxs'], config['constraint_idxs']) 


class PandasLoader(object):
    def __init__(self):
        pass
    

class LhotseLoader(object):
    def __init__(self):
        pass

