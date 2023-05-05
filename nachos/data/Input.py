from nachos.data.Data import Data, Dataset
from itertools import groupby


class TSVLoader(object):
    @staticmethod
    def load(fname, config):
        '''
            Summary:
                Loads the TSV file describing the metadata (factors) and
                converts the metadata into a Dataset object.
                See nachos.data.Data.Dataset for more information.
        '''
        data = []
        with open(fname, 'r') as f:
            headers = f.readline().strip().split('\t')
            # Find the fieldname named fraction. If it does not exist it will
            # be set to 1.
            fraction_idx = None
            # We start with -1, because the first field is assumed to be the id
            # and subsequent factors are then 0 indexed.
            for i, h in enumerate(headers, -1):
                if h.lower() == "fraction":
                    fraction_idx = i 
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
                data.append(
                    Data(
                        record,
                        factors,
                        next(iter(factors[fraction_idx])),
                        field_names=headers[:],
                    )
                )
        return Dataset(data, config['factor_idxs'], config['constraint_idxs']) 


class PandasLoader(object):
    def __init__(self):
        pass
    

class LhotseLoader(object):
    @staticmethod
    def load(supervisions, config):
        '''
            Summary:
                Loads a lhotse supervisions manifests and from them
                creates a Dataset object. See nachos.data.Data.Dataset for more
                information.
        '''
        # First load the lhotse supervisions
        from lhotse import RecordingSet, SupervisionSet
        sups = SupervisionSet.from_segments([])
        supids = set()
        for sup in supervisions:
            new_sups = SupervisionSet.from_jsonl(sup)
            sups = sups + new_sups.filter(lambda s: s.id not in supids)
            for s in new_sups:
                supids.add(s.id)
        
        return Dataset.from_supervisions_and_config(sups, config)
