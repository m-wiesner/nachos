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
        
        # Define the fields to extract from the lhotse manifests
        lhotse_fields = config['lhotse_fields']
        grouping_field = config['lhotse_groupby']
        groups = groupby(sups, lambda s: getattr(s, grouping_field))
        field_names = [grouping_field, *lhotse_fields, 'duration', 'fraction']
        data = []
        factors = {}
        for k, g in groups:
            g_list = list(g)
            factors[k] = [
                set([getattr(s, f) for s in g_list]) for f in lhotse_fields
            ]
            factors[k].append(set([sum(s.duration for s in g_list)])) 
        total_duration = sum(sum(f[-1]) for f in factors.values())
        for k in factors:
            factors[k].append(set([sum(factors[k][-1]) / total_duration]))
            data.append(Data(k, factors[k], field_names=field_names))              
        return Dataset(data, config['factor_idxs'], config['constraint_idxs']) 
