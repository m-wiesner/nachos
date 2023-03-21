from __future__ import annotations
from typing import Any, Optional, List, Dict, Tuple, TYPE_CHECKING
from typing import Generator, Iterable
if TYPE_CHECKING:
    from nachos.similarity_functions.SimilarityFunctions import (
        SimilarityFunctions,
    )
import networkx as nx
import numpy as np
import json
import random


InvertedIndex = Dict[int, Dict[Any, set]] 
Split = Tuple[set, set]
FactoredSplit = Tuple[List[set], List[set]]

class Data(object):
    '''
        Summary:
            A structure to store the factors (including those that will be
            used as constraints) associated with records in a tsv file,
            dataframe, or lhotse manifest.
    '''
    def __init__(self, 
        id: str, factors: list,
        field_names: Optional[list] = None,
    ):
        self.id = id
        self.factors = factors
        # field 0 is the id
        self.field_names = field_names

    def __repr__(self):
        representation = {
            self.id: {
                fieldname: list(factor)
                for fieldname, factor in zip(self.field_names[1:], self.factors)
            }
        }
        return json.dumps(representation).replace(':', '=') 

    def __str__(self):
        return self.__repr__() 


class Dataset(object):
    '''
        Summary:
            A class to store and manipulate the data and their associated
            factors and constraints. The structure we ultimately want is
            similar to an inverted index.
           
            factors = [
                {
                    factor1_value1: [fid1, fid2, ...],
                    factor1_value2: [fids, ...],
                    ...
                },
                {
                    factor2_value1: [...],
                    factor2_value2: [...],
                },
                ...   
            ]
               
    '''
    def __init__(self,
        data: List[Data],
        factor_idxs: List[int],
        constraint_idxs: List[int],
    ):
        self.data = sorted(data, key=lambda x: x.id)
        self.id_to_idx = {x.id: i for i, x in enumerate(self.data)}
        self.factor_idxs = factor_idxs
        self.constraint_idxs = constraint_idxs 
        # Factor, Constraint idxs are 1-indexed so we enumerate starting at 1
        self.factors = {
            d.id: 
                [f for i, f in enumerate(d.factors, 1) if i in factor_idxs]
                for d in self.data
        }
        self.constraints = {
            d.id: 
                [f for i, f in enumerate(d.factors, 1) if i in constraint_idxs]
                for d in self.data
        }
        # Assume all points data are the same type. We could check this but it
        # requires iterating over data, which could take a long time with large
        # data sets. 
        self.field_names = data[0].field_names 
       
        # If requested (graph=True), construct the graph.
        # Sometimes the data may be too large to efficiently construct
        # the graph, but it is useful to check if the graph is complete or
        # to get the number of connected components in the graph.
        self.graph: Optional[nx.Graph] = None
        self.constraint_inv_idx: Optional[InvertedIndex] = None
        self.constraint_values = {n: None for n in range(len(self.constraint_idxs))}
        self.factor_inv_idx: Optional[InvertedIndex] = None
        self.factor_values = {n: None for n in range(len(self.factor_idxs))}

    def subset_from_data(self, d: Iterable[Data]) -> Dataset:  
        '''
            Summary:
                Create a new subset, with the same factors and constraints
                as self, from a subset of the data points.

            Inputs
            ------------------
            :param d: The data points from which to create a Dataset
            :type d: Iterable[Data]

            Returns
            ------------------
            :return: A Dataset object representing the subset of points
            :rtype: Dataset 
        '''
        return Dataset(d, list(self.factor_idxs), list(self.constraint_idxs))

    def subset_from_records(self, r: Iterable[Any]) -> Dataset:
        '''
            Summary:
                Create a new subset, with the same factors and constraints
                as self, from a subset of the data points.
        '''
        pass
         
    def check_complete(self) -> bool:
        '''
            Summary:
                Checks if the graph is complete
                :return: True if complete, False otherwise
                :rtype: bool
        '''
        for n in range(len(self.graph)):
            if self.graph.degree(n) != len(self.graph) - 1:
                return False
        return True

    def check_disconnected(self) -> bool:
        '''
            Summary:
                Checks if the graph if there are M > 1 disconnected components
                in the graph.
            :return: True is disconnected, False otherwise
            :rtype: bool
        '''
        num_components = nx.number_connected_componenets(self.graph)
        self.num_components = num_components
        if num_components > 1:
            return True
        return False 

    def make_graph(self, simfuns: SimilarityFunctions) -> None:
        '''
            Summary:
                Makes the graph representation of the dataset. This assumes
                that the graph is undirected, an assumption which we may later
                break, depending on the kinds of similarity functiosn we will
                ultimately support.
            
            Inputs
            -------------------------------------------
                :param simfuns: the similarity functions (1 per factor) used to
                    compare records (i.e., data points)
                :type simfuns: nachos.SimilarityFunctions.SimilarityFunctions

            Returns
            -------------------------------------------
            :return: returns the graph
            :rtype: numpy.ndarray
        '''
        triu_idxs = np.triu_indices(len(self.data), k=1)
        G = nx.Graph()
        for i, j in zip(triu_idxs[0], triu_idxs[1]):
            # Cast to int to not confuse with np.int64
            i, j = int(i), int(j)
            sim = simfuns(self[i], self[j])
            if sim > 0:
                G.add_edge(i, j, capacity=sim)
        self.graph = G

    def get_record(self, i: int) -> Any:
        return self.data[i].id
    
    def __len__(self) -> int:
        '''
            Summary: 
                Return the lenth of the dataset
            :return: length of the dataset
            :rtype: int
        '''
        return len(self.data)
    
    def __getitem__(self, i: Union[int,slice]) -> Union[Dataset, Data]: 
        '''
            Summary:
                Returns a dataset with a single item (the i-th one).
            
            Inputs
            -------------------------------
            :param i: integer or slice of positions in self.data to select
            
            Returns
            -------------------------------
            :return: returns a dataset with the slice of elements or Data
                with the i-th data element
            :rtype: Dataset, or Data

        '''
        # Make sure to copy the lists so that we don't accidentally modify
        # the original
        if isinstance(i, int):
            return self.subset_from_data([self.data[i]])
        else:
            return self.subset_from_data(self.data[i])

    def export_graph(self, filename) -> None:
        '''
            Summary:
                Exports graph to .gml file which in theory can be read for
                visualization.

            Inputs
            ------------------------------
            :param filename: the filename of the .gml file to create
            :type filename: str
            
            Returns
            -----------------------------
            :return: None
            :rtype: None 
        '''
        if self.graph is not None:
            nx.write_gml(self.graph, filename, stringizer=self.get_record)

    def get_constraints(self, 
        subset: Optional[Iterable] = None, 
        n: Optional[int] = None
    ) -> Generator:
        '''
            Summary:
                Returns a generator over the dataset constraints.
           
            Inputs
            --------------------
            :param subset: Iterable of subset of ids to use
            :type subset: Optional[Iterable] (Default is None) which means use
                all ids.
            :param n: The constraint index to return. By default it is None, 
                which means to return all the constraints.
            :type n: Optional[int]
             
            Returns
            --------------------
            :return: generator over constraints
            :rtype: Generator
        '''
        if subset is None:
            subset = self.constraints.keys()
        keys = set.intersection(set(self.constraints.keys()), set(subset))
        for x in keys:
            yield self.constraints[x] if n is None else self.constraints[x][n]

    def get_factors(self, 
        subset: Optional[Iterable] = None,
        n: Optional[int] = None,
    ) -> Generator:
        '''
            Summary:
                Returns a generator over the dataset factors.
           
            Inputs
            --------------------
            :param subset: Iterable of subset of ids to use
            :type subset: Optional[Iterable] (Default is None) which means use
                all ids.
            :param n: The factor index to return. By default it is None,
                which means to return all factors.
            :type n: Optional[int]
             
            Returns
            --------------------
            :return: generator over factors
            :rtype: Generator
        '''
        if subset is None:
            subset = self.constraints.keys()
        keys = set.intersection(set(self.constraints.keys()), set(subset))
        for x in keys:
            yield self.factors[x] if n is None else self.factors[x][n]

    def make_constraint_inverted_index(self) -> None:
        '''
            Summary:
                Sets the inverted index for the constraints. In other words
                inverted_index[n] = [value1, value2, ...], the set of value
                seen for the n-th constraint.
        '''
        inverted_index = {n: {} for n in range(len(self.constraint_idxs))}
        for fid, x in self.constraints.items():
            for n in range(len(self.constraint_idxs)):
                for y in x[n]:
                    if y not in inverted_index[n]:
                        inverted_index[n][y] = set()
                    inverted_index[n][y].add(fid)
                self.constraint_values[n] = sorted(inverted_index[n])
        self.constraint_inv_idx = inverted_index
    
    def make_factor_inverted_index(self) -> None:
        '''
            Summary:
                Returns the inverted index for the factors. In other words
                inverted_index[n] = [value1, value2, ...], the set of value
                seen for the n-th factor.
        '''
        inverted_index = {n: {} for n in range(len(self.factor_idxs))}
        for fid, x in self.factors.items():
            for n in range(len(self.factor_idxs)):
                for y in x[n]:
                    if y not in inverted_index[n]:
                        inverted_index[n][y] = set()
                    inverted_index[n][y].add(fid)
                self.factor_values[n] = sorted(inverted_index[n].keys())
        self.factor_inv_idx = inverted_index

    def draw_random_split_from_factor(self, n: int) -> Tuple[int, Split]:
        '''
            Summary:
                Return a set of Data point ids and its complement corresponding
                to the inclusion of a subset of values selected from the n-th
                factor into the "training" set. We also return the index of the
                set from the powerset of values that resulted in the split.

            Inputs
            -----------------
            :param n: the index of the factor in the list self.factor_idxs from
                which to select
            :type n: int

            Returns
            ------------------
            :return: The tuple of the index of the set from the powerset of
                values and the datasets corresponding to the random split
                and it's complement resulting from that index
            :rtype: Tuple[int, Tuple[set, set]]
        '''
        if self.factor_inv_idx is None:
            self.make_factor_inverted_index()

        factor_values = self.factor_inv_idx[n]
        # To select a random subset we will select a random subsets from the
        # powerset of values. Each bit in the binary representation of the index
        # of one of these subsets can be interpretted as the presence of a 
        # specific factor value in the selected subset. We don't want to select
        # the emptyset or the full set of values because this puts all of the
        # data into a single set, and we want to form both training and test
        # partitions. This is why the random in random.randit() is from 1 to
        # 2**len(factor_values) - 2.
        subset_idx = random.randint(1, 2**len(factor_values) - 2)
        subset_idx_binary = format(subset_idx, f'0{len(factor_values)}b')

        include, exclude = [], []
        for i, j in enumerate(subset_idx_binary):
            if j == '1':
                include.append(
                    factor_values[self.factor_values[n][int(i)]]
                )
            elif j == '0':
                exclude.append(
                    factor_values[self.factor_values[n][int(i)]]
                )
        subset_from_selected_factors = set().union(*include)
        subset_from_unselected_factors = set().union(*exclude)

        # For multilabel problems, there can be overlap between the data points
        # that have selected factors and unselected factors. We need to find
        # this intersection.
        intersection = subset_from_selected_factors.intersection(
            subset_from_unselected_factors
        )
        
        # We remove the intersection from both sets
        subset = subset_from_selected_factors.difference(intersection)
        
        # not_subset is our name for the complement of subset.
        not_subset = subset_from_unselected_factors.difference(intersection)

        return (subset_idx, (subset, not_subset,))

    def draw_random_split(self) -> Tuple[List[int], FactoredSplit]:
        '''
            Summary:
                Applies self.draw_random_split_from_factor() to each factor
                independently, and returns all of the splits.

            Returns
            ------------------------
            :return: The keys (indices into the powersets of values for each
                factor), and the values (the selected Dataset and its complement)
                for each factor.
            :rtype: Tuple[List[int], List[Tuple[Dataset, Dataset]]]
        '''
        subsets, not_subsets = [], []
        indices = []
        for n in range(len(self.factor_idxs)):
            idx_n, split = self.draw_random_split_from_factor(n)
            subsets.append(split[0])
            not_subsets.append(split[1])
            indices.append(idx_n)

        return (indices, (subsets, not_subsets))
    
    def set_random_seed(self, seed: int = 0) -> None:
        '''
            Summary:
                Set the random seed of the random module

            Inputs
            -----------------
            :param seed: Default to 0. It's the random module's random seed
            :type seed: int
        '''
        random.seed(seed)


def collapse_factored_split(split: FactoredSplit) -> Split:
    '''
        Summary:
            Take a FactoredSplit and collapse it by intersecting all the
            selected set, and intersecting all of their complements to create
            a single selected set and a single other split with no overlap in
            any of the factors present in the selected set.
    
        Inputs
        -----------------------
        :param split: The split to collapse
        :type split: FactoredSplit

        Returns
        ------------------------
        :return: the collapsed split
        :rtype: Split
    '''
    return (set.intersection(*split[0]), set.intersection(*split[1]))
