from __future__ import annotations
from typing import Any, Optional, List, Dict, Tuple, TYPE_CHECKING
from typing import Generator, Iterable
if TYPE_CHECKING:
    from nachos.similarity_functions.SimilarityFunctions import (
        SimilarityFunctions,
    )
import networkx as nx
from networkx.algorithms.connectivity import (
    build_auxiliary_node_connectivity,
    minimum_st_node_cut,
)
from networkx.algorithms.flow import build_residual_network

import numpy as np
import json
import random
from itertools import groupby
from copy import deepcopy
from tqdm import tqdm


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

    def copy(self) -> Data:
        factors = self.factors[:]
        return Data(self.id, factors, field_names=self.field_names)


class Dataset(object):
    """Summary:
            A class to store and manipulate the data and their associated
            factors and constraints. The structure we ultimately want is
            similar to an inverted index. This inverted index is effectively
            stored when we create factor specific graphs when 

            self.make_graph() 

            is called.
        :param data: the list of Data classes storing the data instances
        :type data: List[Data]
        :param factor_idxs: the list of integers specifying which fields in
            data should be treated as factors
        :type factor_idxs: List[int]
        :param constraint_idxs: the list of integers specifying which fields
            in data should be treated as constraints
        :type constraint_idxs: List[int] 
    """
    @classmethod
    def from_supervisions_and_config(cls, supervisions, config):
        '''
            Summary:
                Creates a dataset from lhotse supervisions.

            :param supervisions: The lhotse supervisions
            :type supervisions: SupervisionSet (defined in lhotse)
            :param config: Path to a configurations file
            :type config: str
            :return: A dataset class from the supervisions
            :rtype: Dataset
        '''
        # Define the fields to extract from the lhotse manifests
        lhotse_fields = config['lhotse_fields']
        grouping_field = config['lhotse_groupby']
        groups = groupby(supervisions, lambda s: getattr(s, grouping_field))
        field_names = [grouping_field, *lhotse_fields, 'duration', 'fraction']
        return cls.from_supervisions(
            supervisions,
            lhotse_fields,
            config['factor_idxs'],
            config['constraint_idxs'],
            groupby_field=grouping_field,
        )
           
    @classmethod
    def from_supervisions(cls, 
        supervisions,
        fields: List[str],
        factor_idxs: List[int],
        constraint_idxs: List[int],
        groupby_field: str = 'recording_id',
    ):
        '''
            Summary:
                Creates a dataset from lhotse supervisions.

            :param supervisions: The lhotse supervisions
            :type supervisions: SupervisionSet (defined in lhotse)
            :param fields: A list of the fieldnames to extract from the lhotse
                supervisions
            :type fields: List[str]
            :param factor_idxs: A list of integers specifying which of the
                fields to use as factors.
            :type factor_idxs: List[int]
            :param constraint_idxs: A list of integers specifying which of the
                fields to use as constraints.
            :type constraint_idxs: List[int]
            :type config: str
            :return: A dataset class from the supervisions
            :rtype: Dataset
        '''
        # Define the fields to extract from the lhotse manifests
        groups = groupby(supervisions, lambda s: getattr(s, groupby_field))
        field_names = [groupby_field, *fields, 'duration', 'fraction']
        data = []
        factors = {}
        for k, g in groups:
            g_list = list(g)
            factors[k] = [
                set([getattr(s, f) for s in g_list]) for f in fields
            ]
            factors[k].append(set([sum(s.duration for s in g_list)])) 
        total_duration = sum(sum(f[-1]) for f in factors.values())
        for k in factors:
            factors[k].append(set([sum(factors[k][-1]) / total_duration]))
            data.append(Data(k, factors[k], field_names=field_names))     
        return cls(data, factor_idxs, constraint_idxs)

    
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
                [d.factors[i-1] for i in constraint_idxs]
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
        self.graphs = None
        self.A = None # Auxiliary node connectivity graph
        self.R = None # The residual node connectivity graph

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


    def is_complete(self) -> bool:
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

    def is_disconnected(self) -> bool:
        '''
            Summary:
                Checks if the graph if there are M > 1 disconnected components
                in the graph.
            :return: True is disconnected, False otherwise
            :rtype: bool
        '''
        if self.graph is None:
            raise RuntimeError("Cannot check if graph is disconnected when it "
                "has not yet been created. Create the graph by calling "
                "self.make_graph(sim_fn)."
            )
        num_components = len(list(nx.connected_components(self.graph)))
        self.num_components = num_components
        if num_components > 1:
            print(f'Num Components: {num_components}')
            return True
        return False

    def make_graph(self, simfuns: SimilarityFunctions) -> None:
        '''
            Summary:
                Makes the graph representation of the dataset. This assumes
                that the graph is undirected, an assumption which we may later
                break, depending on the kinds of similarity functiosn we will
                ultimately support. It also makes subgraphs corresponding to
                each individual factor value. This is like the inverted index.
                You can lookup the neighbors of a factors. The graph and
                factors are stored in self.graph and self.graphs respectively.

            Inputs
            -------------------------------------------
                :param simfuns: the similarity functions (1 per factor) used to
                    compare records (i.e., data points)
                :type simfuns: nachos.SimilarityFunctions.SimilarityFunctions

        '''
        triu_idxs = np.triu_indices(len(self.data), k=1)
        self.graphs = {i: {} for i in range(len(self.factor_idxs))}
        G = nx.Graph()
        G.add_nodes_from(range(len(self.data)))
        for i, j in tqdm(zip(triu_idxs[0], triu_idxs[1]), desc="Making graph", total=len(triu_idxs[0])):
            # Cast to int to not confuse with np.int64
            i, j = int(i), int(j)
            # Loop over each factor to store inverted index of factor to graph
            # of data points with similar factors
            for n in range(len(self.factor_idxs)):
                # factor_idxs is 1-indexed, so the factor to take is at
                # position n-1 in self.data[i].factors for the n-th factor in
                # factor_idxs
                factor_i = self.data[i].factors[self.factor_idxs[n]-1]
                factor_j = self.data[j].factors[self.factor_idxs[n]-1]
                # Since factors can be multivalued we need to loop over each
                # value for a specific factor type, i.e., each speaker present
                # in a recording. We instantiate the graph if a particular value
                # has not been seen.

                # Instantiating the graph for any unseen values in the n-th
                # factor of the i-th data point
                for f in factor_i:
                    if f not in self.graphs[n]:
                        self.graphs[n][f] = nx.Graph()
                # Instantiating the graph for any unseen values in the n-th
                # factor of the j-th data point
                for f in factor_j:
                    if f not in self.graphs[n]:
                        self.graphs[n][f] = nx.Graph()
                # Now we need to find the similar data points for each factor.
                # Because factor values can be multivalued we need to loop
                # through each value an compute similarity of a datapoints
                # with respect to each factor separately.
                for f in factor_i:
                    u_i = self[i]
                    u_i.data = u_i.data.copy()

                    # Create a dummy Dataset with a single datapont and a
                    # single value for the the n-th factor of the one one point
                    u_i.data[0].factors[n] = {f}
                    # Now compute the similarity with respect to that one point
                    # and add an edge to the value specific graph of the n-th
                    # factor if the similarity is > 0
                    self.graphs[n][f].add_node(i)
                    sim = simfuns(self[j], u_i, n=n)
                    if sim > 0:
                        self.graphs[n][f].add_edge(i, j, capacity=sim)
                # Repeat the whole above process but for factor_j
                for f in factor_j:
                    u_j = self[j]
                    u_j.data = u_j.data.copy()
                    # Create a dummy Dataset with a single datapont and a
                    # single value for the the n-th factor of the one one point
                    u_j.data[0].factors[n] = {f}
                    # Now compute the similarity with respect to that one point
                    # and add an edge to the value specific graph of the n-th
                    # factor if the similarity is > 0
                    self.graphs[n][f].add_node(j)
                    sim = simfuns(self[i], u_j, n=n)
                    if sim > 0:
                        self.graphs[n][f].add_edge(i, j, capacity=sim)

            # Now compute the normal similarity (summing the score across all
            # factors), and creat the edge in the global graph (that includes
            # all the data points.
            sim = simfuns(self[i], self[j])
            if sim > 0:
                G.add_edge(i, j, capacity=sim)
        self.graph = G
        for n in range(len(self.factor_idxs)):
            self.factor_values[n] = sorted(self.graphs[n].keys())

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
            return self.subset_from_data([self.data[i].copy()])
        else:
            return self.subset_from_data(self.data[i].copy())

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
                seen for the n-th factor. This is really not a particularly
                useful function, as the inverted index computed in this way
                only works for the set_intersection similarity method. For other
                types of similarity, such as cosine distance, self.make_graph()
                will make a the graphs corresponding to each factor, and is
                really a better version of the inverted index created in this
                function.

                This function therefore exists mostly to mirror what the
                make_cosntraint_inverted_index function.
        '''
        # We only need to run this if it has not yet been computed. Otherwise
        # just skip this.
        if self.factor_inv_idx is not None:
            return
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
        # To select a random subset we will select a random subsets from the
        # powerset of values. Each bit in the binary representation of the index
        # of one of these subsets can be interpretted as the presence of a
        # specific factor value in the selected subset. We don't want to select
        # the emptyset or the full set of values because this puts all of the
        # data into a single set, and we want to form both training and test
        # partitions. This is why the random in random.randit() is from 1 to
        # 2**len(factor_graphs) - 2.
        subset_idx = random.randint(1, 2**len(self.graphs[n]) - 2)
        return self.draw_split_from_factor(n, subset_idx)

    def draw_split_from_factor(self, n: int, idx: int) -> Tuple[int, Split]:
        '''
            Summary:
                Like draw_random_split from factor, but draws the split
                specified by an integer index, idx, which specifies the subset
                of values from the powerset of values from factor n to use.

            Inputs
            -----------------
            :param n: the index of the factor in the list self.factor_idxs from
                which to select
            :type n: int
            :param idx: The index in the powerset of the subset of values from
                the n-th factor to use.
            :type idx: int

            Returns
            ------------------
            :return: The tuple of the index of the set from the powerset of
                values and the datasets corresponding to the random split
                and it's complement resulting from that index
            :rtype: Tuple[int, Tuple[set, set]]
        '''
        if self.graphs is None:
            self.make_graph()

        # Get the list of graphs (one per factor value) associated with the
        # n-th factor.
        factor_graphs = self.graphs[n]

        subset_idx = idx
        subset_idx_binary = format(subset_idx, f'0{len(factor_graphs)}b')

        include, exclude = [], []
        for i, j in enumerate(subset_idx_binary):
            if j == '1':
                key = self.factor_values[n][int(i)]
                include.append(
                    [self.data[i].id for i in factor_graphs[key].nodes]
                )
            elif j == '0':
                key = self.factor_values[n][int(i)]
                exclude.append(
                    [self.data[i].id for i in factor_graphs[key].nodes]
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

    def nearby_splits(self, idxs: List[int], split: FactoredSplit) -> Generator[Tuple[List[int], FactoredSplit]]:
        '''
            Summary:
                Make a generator over "neaby splits". These are splits that
                are Hamming distance 1 away from the current split. By this we
                mean if you concatenated the bit strings representing the
                indices of the powersets of values for each factor, then any
                bit string that differs in a single value.

            Inputs
            --------------------
            :param idx: The indices into the powersets of the subset
                corresponding to split
            :type idxs: List[int]
            :param split: a split (a factored split actually) around which we
                want to find splits that are Hamming distance = 1 away
            :type split: FactoredSplit

            Returns
            -----------------------
            :return: a generator over the neighboring splits
            :rtype: Generator[FactoredSplit]
        '''
        # We want to randomize the order of the factors that we are exploring
        factor_order = list(range(len(self.factor_idxs)))
        random.shuffle(factor_order)
        for i in factor_order:
            v_binary = format(idxs[i], f'0{len(self.graphs[i])}b')
            for n in _1bit_different_numbers(v_binary):
                subset_idx = int(n, 2)
                new_idx, new_split = self.draw_split_from_factor(i, subset_idx)
                indices = [idxs[j] if j != i else new_idx for j in range(len(self.factor_idxs))]
                subsets = [
                    split[0][j] if j != i else new_split[0]
                    for j in range(len(self.factor_idxs))
                ]
                not_subsets = [
                    split[1][j] if j != i else new_split[1]
                    for j in range(len(split[0]))
                ]
                yield (indices, (subsets, not_subsets))

    def get_neighborhood(self,
        idxs: List[int],
        split: FactoredSplit,
        l: int,
        max_neighbors: int = 2000,
    ) -> Generator[Tuple[List[int], FactoredSplit]]:
            '''
                Summary:
                    Return a generator over all of the neighbors at distance l from
                    split.

                Inputs
                --------------------
                :param idxs: The list of indices, for each factor into their
                    respective powersets of the corresponding to the splits
                :type idxs: List[int]
                :param split: The split whose neighbors at distance l we want to
                    generate.
                :type split: FactoredSplit
                :param l: The distance from split of the neighbors we would like to
                    generate
                :type l: int
                :param max_neighbors: The maximum number of neighbors to explore
                :type max_neighbors: int

                Returns
                --------------------
                :return: A generator over the neighbors at distance l from split
                :rtype: Generator[FactoredSplit]
            '''
            # Initialize neighbors.
            neighbors = list(
                map(lambda x: (x, 1), self.nearby_splits(idxs, split))
            )
            # For each neighbor we want to retreive all of their neighbors and
            # repeat this process until we have explored up to the l-th degree of
            # connections. Most of this while loop is just adding neighbors to
            # explore to the list of neighbors we already need to explore. The
            # base case (under the else statement) is where the neighbors actually
            # are yielded.
            num_neighbors = 0
            while len(neighbors) > 0 and num_neighbors < max_neighbors:
                (nearby_idxs, nearby_split), idx_l = neighbors.pop()
                if idx_l < l:
                    for x in self.nearby_splits(nearby_idxs, nearby_split):
                        neighbors.append((x, idx_l + 1))
                else:
                    num_neighbors += 1
                    yield (nearby_idxs, nearby_split)

    def shake(self, idxs: List[int], split: FactoredSplit, k: int) -> Tuple[int, FactoredSplit]:
        '''
            Summary:
                Return a random split from the neighborhood around split.

            Inputs
            --------------------
            :param idx: The index o
            :param split: The current split around which we will select a random
                neighbor
            :type split: FactoredSplit
            :param k: The distance from the split form which our new split,
                obtained by shaking will be drawn from. Kind of like a shake
                distance
            :type k: int

            Return
            ------------------------
            :return: The randomly selected split from the neighborhood of split
            :rtype: FactoredSplit
        '''
        generator = self.get_neighborhood(idxs, split, k)
        return next(generator, None)

    def draw_random_node_cut(self) -> Split:
        '''
            Summary:
                Draw random, non-adjacenent verticies as source and target
                nodes, and compute the minimum st-vertex cut. This cut may
                result in > 2 components. In this case, randomly assign the
                components to different splits.

            Returns
            -------------------------
            :return: the split of components
            :rtype: Split
        '''
        if self.A is not None:
            self.A = build_auxiliary_node_connectivity(d.graph)
            self.R = build_residual_network(A, 'capicity')

        nodes_are_adjacent = True
        while nodes_are_adjacent:
            sampled_nodes = random.sample(range(len(self.graph)), 2)
            if sampled_nodes[1] not in self.graph.neighbors(sampled_nodes[0]):
                nodes_are_adjacent = False

        cut = minimum_st_node_cut(
            self.graph, sampled_nodes[0], sampled_nodes[1],
            auxiliary=self.A, residual=self.R,
        )
        H = deepcopy(self.graph)
        for n in cut:
            H.remove_node(n)
        components = sorted(nx.connected_components(H), key=len, reverse=True)
        num_components = len(components)
        sequence = list(range(num_components))
        random.shuffle(sequence)
        include, exclude = [], []
        # Assign component (if there are > 2) to include and exclude
        for i, component_idx in enumerate(sequence):
            component_ids = [self.data[j].id for j in components[component_idx]]
            # Just make the last component part of exclude
            if i == num_components - 1:
                exclude.append(component_ids)
                continue
            if i % 2 == 0:
                include.append(component_ids)
                continue
            if i % 2 == 1:
                exclude.append(component_ids)

        subset = set().union(*include)
        not_subset = set().union(*exclude)
        return (subset, not_subset)


    def make_overlapping_test_sets(self, split: Split) -> List[set]:
        '''
            Summary:
                Takes a split of the dataset, i.e., two subsets of the dataset
                that do not overlap in the specified factors, and from the
                remaining data in the dataset not included in the split, creates
                multiple test sets that have some overlap with respect to
                one or more factors in the first of the two subsets in split.

                In general, there are 2^N different kinds of overlap when using
                N factors. By overlap, we mean factors that are considered under
                the similarity function used to create the graph. We can use
                the factors specific graphs for this purpose.

            Inputs
            ------------------------
            :param split: The split (i.e., two subsets of the data sets) with
                no factor overlap with respect to which we are making the
                additional test sets.
            :type split: Split

            Returns
            ------------------------------
            :return: Test sets
            :rtype: List[set]
        '''
        N = len(self.factor_idxs)
        # We first want to retrieve all data points that overlap with factors
        # that were used in split[0]
        data_w_overlapping_factors = {
                i: [] for i in range(len(self.factor_idxs))
        }
        
        for key in split[0]:
            for f_idx, factor in enumerate(self.factors[key]):
                for f in factor:
                    data_w_overlapping_factors[f_idx].append(
                        [self.data[i].id for i in self.graphs[f_idx][f].nodes]
                    )

        # We also want to find all of the data points that were not used
        # in the split
        unused_indices = [
            i.id for i in self.data
                if i.id not in split[0] and i.id not in split[1]
        ]

        # Once we have all the used factors and unused data points we need to
        # create splits containing overlap in only some of the factors
        subsets = {}
        for idx in range(2**N):
            # Create the binary representation for the kind of overlapping test
            # set to generate. e.g., Overlapping in factor 0, but
            # non-overlapping in factor 1 would be "01".
            subset_binary = format(idx, f'0{N}b')
            new_set = set(unused_indices)
            for i, j in enumerate(subset_binary): #(ABC) 011 --> A and !B and !C
                overlapping_set = set().union(*data_w_overlapping_factors[i])
                if j == '0':
                    new_set = new_set.intersection(overlapping_set)
                elif j == '1':
                    new_set = new_set.intersection(
                        set(unused_indices).difference(overlapping_set)
                    )
                subsets[idx] = new_set
        return subsets

    def overlap_stats(self, s1: set, s2: set) -> dict:
        '''
            Summary:
                Compute the overlap s2 w/r s1 "stats" associated with each
                factor.

            Inputs
            --------------------------
            :param s1: The set with respect to which overlap will be computed
            :type s1: set
            :param s2: the set whose overlap is computed with respect to s1
            :type s2: set

            Returns
            -----------------------------
            :return: The dictionary of factors overlaps (s2 w/r to s1)
            :rtype: dict
        '''
        data_w_overlapping_factors = {
                i: [] for i in range(len(self.factor_idxs))
        }
        overlap_stats = {}
        # Find the fraction of points in s1 that are overlapping w/r to s2
        # in each factor. To do this we need to gather all the values for
        # a specific fator in s2, and examine all the similar points to those
        # factor values. Intersecting these points with the points from s2
        # gives the fraction of points that are overlapped w / r to that factor
        for d1 in s1:
            for f_idx, factor in enumerate(self.factors[d1]):
                for f in factor:
                    data_w_overlapping_factors[f_idx].append(
                        [self.data[i].id for i in self.graphs[f_idx][f].nodes]
                    )
                overlapping_points = s2.intersection(
                    set().union(*data_w_overlapping_factors[f_idx])
                )
                field_name = self.field_names[self.factor_idxs[f_idx]]
                overlap_stats[field_name] = round(
                    len(overlapping_points) / len(s2), 4
                )
        return overlap_stats

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


def _1bit_different_numbers(v: str) -> Generator[str]:
        '''
            Summary:
                From a bit string v representing an index find all of the bit
                strings (i.e. integrers) that differ by only 1 bit.
        '''
        # Special edge case for len(v) == 2. We need to use a 2-bit difference
        # since otherwise, the empty set or full set will be returned and we
        # do not want to include those.
        if v == '01':
            yield '10'
        elif v == '10':
            yield '01'

        sequence = list(range(len(v)))
        random.shuffle(sequence)
        for i in sequence:
            new_val = list(v)
            new_val[i] = '1' if v[i] == '0' else '0'
            g = groupby(new_val, lambda x: x)
            if not (next(g, True) and not next(g, False)):
                yield ''.join(new_val)

def new_from_components(d: Dataset,
    simfuns: SimilarityFunctions,
) -> Tuple[Dataset, List[set]]:
    '''
        Summary:
            Create a new dataset from the existing one by checking if > 1
            connected components exist, and using the components as the
            records to split rather than the original data points. In this
            case there are no factors to consider (other than the
            trivial factor which is just the component ID), and just
            constraints.
        :param d: The dataset from which to generate a new one using its
            components
        :type d: Dataset 
        :param simfuns: The similarity functions used on each of the
            originally specified factors to create the graph
        :type simfuns: SimilarityFunctions
        :return: A new dataset constructed from the connected components
            of the original along with the map from the new points to the
            corresponding set of old points.
        :rtype: Tuple[Dataset, List[set]]
    '''
    if d.graph is None:
        d.make_graph(simfuns)
    components = list(nx.connected_components(d.graph))
    if len(components) > 1:
        # Restructure dataset to use clusters as the factor
        # get id to component map
        data = []
        id_to_component = {}
        for c_idx, c in enumerate(components):
            new_fields = []
            for i in range(len(d.data[0].factors)):
                new_fields.append([f for j in c for f in d.data[j].factors[i]])
            new_fields.append({c_idx})
            for data_idx in c:
                data_id = d.get_record(data_idx)
                id_to_component[data_id] = c_idx
            new_field_names = d.field_names[:]
            new_field_names.append('component')
            data.append(
                Data(c_idx, new_fields, field_names=new_field_names)
            )
        factor_idxs = [len(data[0].factors)]
        constraint_idxs = d.constraint_idxs[:]
        dataset = Dataset(data, factor_idxs, constraint_idxs)
        from nachos.similarity_functions.SimilarityFunctions import (
            SimilarityFunctions,
        )
        from nachos.similarity_functions.boolean import Boolean
        sim_fns = SimilarityFunctions([Boolean()], [1.0])
        dataset.make_graph(sim_fns)
        components_ = [set([d.get_record(i) for i in c]) for c in components]
        return dataset, components_ #id_to_component 
    else:
        return d, [set(d.factors.keys())] #{d_i.id: d_i.id for d_i in d}

