# NACHOS — Nearly Automatic Creation of Held Out Splits
nachos is a python library designed to automatically partition datasets into
splits useable for training and testing machine learning algorithms. 

## About
Machine learning algorithms rely on data for parameter estimation and model
evaluation. Typically datasets used for parameter estimation and model
evaluation are derived by splitting available data into disjoint sets — the
train and development / evaluation sets. 

The authors of this toolkit began noticing that machine learning practitioners
do not have any standard tools for creating these splits, and, with the
proliferation of end-to-end neural models that attempt to jointly model
the production of desired outputs directly from raw inputs, many released
data splits, including commonly used splits, are not designed to test model
generalization in meaningful ways. In some community-wide speech recognition
challenges, the provided test/development sets were drawn from speakers **not**
seen in the training set but documents that **were** seen in the training set.

Speech recognition practitioners have also repurposed dataset splits, such as those defined in Librispeech,
that were specifically designed for Hybrid ASR models. However the dev and test sets contain
signficant overlap of source material with examples in the training set.
Therefore, evaluation of end-to-end models trained and evaluated using these splits
may not yield meaningful results.

This toolkit is a python library designed to provide formal, reproducible, and
"correct" methods for splitting datasets to avoid these problems. 

## Goals
- Become the de facto method for creating data partitions for machine learning
- Formalize the assumptions made when creating data partitions so that the
  hypotheses tested when evaluating on specific splits are clear
- Interface seamlessly with other tools for data preparation, especially audio
  data preparation tools such as lhotse.
- Design stress tests of machine learning models by enabling splits containing
  only, e.g., the least similar speakers or text from those seen in training. 
- Through the use of properly created splits, potentially we will explore
  how to use such splits for causal reasoning.

## Example usage

To come ...

## Core Concepts

The core idea behind this splitting tool is to model data as a graph. Each
vertex in the graph represents a unit of data — be it a sentence, a recording,
a group of sentences uttered by a speaker, or a group of utterances
consisting of the same sentence uttered by different speakers — and edges
between vertices indicate that those units of data are similar to each other.
By similar, we mean that there exists some underlying factor that relates the
two vertices. For instance two vertices might correspond to the same, or
similar speakers, the same sentence, were drawn from the same book, correspond
to the same accent, etc.. Splits are created specifically to test
generalization to new instances of these latent factors.  The task of creating
a splits then becomes to either find, or create disconnected components in the
graph. **Factors** are the properties of the data for which we wish to test
generalization.

<img src="nachos_graphs.png"  width="50%" height="50%">

In general, we may wish for these disconnected components to also have certain
properties: they should be close to a specified size, they should have similar
distributions of speakers, genders, or perhaps durations. We call these kinds of
properties **constraints**.

nachos defines different kinds of **splitters** which operate on datasets. They
split datasets, which are represented by a **Dataset** class, in two, creating
a training set, and a heldout set. The heldout set is special since it has been
selected specifically to be disjoint from the training set in all of the
specified **factors** and has also been chosen to at least approximately satisfy
any specified **constraints**.

A **Dataset** is simply a container for a list of data points, along with a set
of factors and constraints.

A splitter operates on the Dataset by inducing a specific graph using
**SimilarityFunctions** and **Constraints**. These are classes that group
together relationships between and constraints on each field associated with a
data point. For instance, a speaker similarity function could operate on a
speaker field associated with a data point and a set similarity function could
operate on the set of prompts that a specific speaker reads. The constraints
could be functions to ensure that the training and test partitions have matcehd
gender, and are of specific sizes.

<img src="nachos_structure.png"  width="40%" height="40%">

In general a splitter splits the data into 3 portions. The first portion can be
used as a training set. The second portion can be used as a test set. The 
remaining data, i.e., the third portion is all of the remaining data which may
have some amount of overlap in some factor with the training data, or its
inclusions caused the training and test sets to deviate too far from the
desired constraints. That doesn't mean that the data cannot be used.

In nachos, we automatically partition the remaining data in all of the
sets that have overlap, or not, with respect to each factors. These datasets
could then be used to test generalization to specific factors, or sets of
factors rather than to all of them simultanesouly. The caveat with these sets
is that we generally have no control over their sizes, so some may be very small,
and furthermore, any other constraints we imposed on our splits were specifically
not applied to the remaining data.

## Installation
```
git clone https://github.com/m-wiesner/nachos.git
cd nachos 
pip install -r requirements.txt
pip install -e . 
```
=======
# NACHOS -- Nearly Automatic Creation of Held Out Splits

This toolkit provide several methods for creating heldout splits using a 
file containing metadata about the units over which we are splitting. These
metadata may include features such as the speaker(s) present in a recording,
the gender of the speaker, the duration of the recording, the prompt spoken, 
the room in which it was spoken etc...

>>>>>>> 1a895612c78af0647a624aeb63e87a614db45b32
## What you need to do

Create a file representing your corpus that has a column for the id of each
element of the corpus (i.e., the one you are trying to split), and then a
<<<<<<< HEAD
column for each field of metadata you have about each element in the corpus.
These metadata could be prompt, speaker ID, accent, age, health status, duration, ...

Such a file might look like the following

|id|spks|room|subj_gender|intv_gender|data_fraction|
|---|---|---|---|---|---|
|ID1|s1|r1|0|0|0.002|
|ID2|s2,s3|r1|1|0|0.00102|
|ID3|s1,s3|r2|1|0|0.005|
|ID4|s2|r2|1|1|0.02|
|ID5|s1|r3|1|0|0.00223|
|ID6|s4|r1|1|1|0.0042|
|ID7|s2,s4|r2|1|0|0.1|

This file should be a tab-separated (.tsv) file that represents the metadata
associated with your dataset.

## Running

Set the values in config.yaml to the desired values and then

```
# Lhotse manifests
python run.py test/fixtures/config.yaml test/fixtures/supervisions_train_intv.jsonl.gz test/fixtures/supervisions_dev_a.jsonl.gz

# or

# TSV file
python run.py test/fixtures/connected_test_constraints.yaml test/fixtures/connected_fraction_constraints.tsv

# In general
python run.py config metadata_file
```

## Options
Running nachos depends on the following 8. steps/choices:
1. Specification of metadata
2. The grouping of the data into splittable units:
  - recordings
  - all utterances uttered by a speaker
  - utterances
  - chapters (from audiobooks)
  - documents
  - utterances of a specific speaker in a specific document
  - and many more ...
3. Chosing which factors to isolate when creating the train/test split
4. Chosing which constraints to enforce when creating the train/test split
5. Defining the similarity functions that operate on the specified factors fields of the metadata
6. Defining the constraints to enforce on the specified constraint fields of the metadata
7. Defining the splitter to use to creating the train/test split
8. Possibly adjust some hyperparameters and run different random seeds to obtain better splits.

### Steps 1-6.: Metadata, Similarity, and Constraints
In general, the values specified for the metadata will decide what kind of similarity and constraint funtions to use. For instance, if I have specified that I would like a speaker disjoint train/test split, and the speaker field takes sets of speakers as values, then the obvious similarity function to use for that factor is the **set_intersection** function, which measures the overlap of two sets. So if recording #1 has speakers {s1, s2, s3} and recording #2 has speakers {s1, s3, s4}, the overlap between these sets would be 2.

If on the other hand the metadata were specified in terms of x-vectors for each speaker, i.e., vectors $s_1, s_2$, then the similarity used might be the thresholded **cosine** similarity. In this case, the similarity would be 

$$s_1^Ts_2$$ 

assuming the x-vectors were normalized to have unit magnitude. The threshold is used to control the number of edges in the graph. If too small a value is used, the graph is either a complete graph or very close to a complete graph

Similarly the values for the constraints can make the choice of constraint function obvious. If each utterance is associated with a label, such as a topic, one might want to create a split where the distribution over topics in the training and test sets is matched. This can be acheived by using the kl constraint on the topic field.

### Step 7.: Splitters
The specified metadata will also generally define the kind of splitter that is optimal to use. The structure of the graph induced by the metadata will generally dictate which splitter to use: a complete graph can not be split into train/test partition with disjoint factors. Instead we can minimize the overlap of factors using minimum-edge-cuts. Normally in this situation, we will use the **spectral_clustering** splitter and specify the number of clusters to return. These clusters are generally of similar size, and they can then be grouped to create splits of a desired size.

When the graph is connected (but not complete), we recommend using the **vns** (variable neighborhood search) splitter. This is an approximate search algorithm that searches over a space of feasible solutions for solutions that best satisfy the specified constraints. Actually, the algorithm can also be used if the graph is disconnected. If there are not many constraints to satisfy and the primary consideration is to use as much data as possible, the **min_node_cut** splitter is probably the best choice. This splitter works similarly to the **vns** splitter, but searches over a more restricted set of feasible solutions: those that are minimum vertex cuts of the graph. It may be difficult with this splitter to approximate the specified constraints (size or otherwise) because the search space is so restricted. However, it will likely result in the best data efficiency; i.e., most data will be used either in the training or test splits and the minimum amount will be discarded due to overlap. 

### Step 8.: Hyperparameter tuning
In the config.yaml file, there are many default parameters corresponding to the different splitter, similarity and constraint functions. They were set to perform well on specific datasets that we have tried in the past, balancing speed of the splitters with the closeness of fit to the desired constraints. However, changing these values may result in a better splits on new datasets. We especially recommend playing with the **constraint_weights** hyperparameter and the random seeds. Kicking off multiple instances of the splitter with different random seeds is an easy, and hyperparameter-free method of searching over more potential splits. Many statistics about the returned splits are printed and can be inspected when selected which split to use. They look like the following.

```
Split 0: {'fraction': 0.7728, 'length': 182}
Split 1: {'fraction': 0.1997, 'length': 58}
Split 2: length = 0
Split 3: {'fraction': 0.0275, 'length': 8}
1 overlap with 0: {'speaker': 0.0}
3 overlap with 0: {'speaker': 0.0}
0 overlap with 1: {'speaker': 0.0}
3 overlap with 1: {'speaker': 0.0}
0 overlap with 3: {'speaker': 0.0}
1 overlap with 3: {'speaker': 0.0}
```

Split 0 is by definition the selected set, and Split 1 is the "heldout" set with respect to Split 0. The remaining splits come from the discarded data and have different kinds of overlapping factors with respect to Spit 0. The values of the constraints for split are printed along with the number of data units in each split. Finally, the pairwise overlap between splits with respect to each factor is printed to screen. In the above example, it so happens that the factor (speaker) is not overlapped between any of the splits. The constraint, which was on the fraction of data included in each split (we used the **sum_tuple** constraint on the field called fraction, which had the fraction of data contained within each unit), is printed. We requested 80% of the data in Split 0 and 20% in Split 1, which the algorithm nearly managed to accomplish. We set the parameter **sum_tuple** in the yaml config file.

```
sum_tuple: [0.8, 0.2]
```

## A Note of Caution
Some of these algorithms currently naively construct the dataset graph, which can take a long time even for modest numbers of "records", i.e., the vertices in the graph, or units of data that we are splitting. One future update will be to make some of the graph processing more scalable. The search algorithms can also become fairly slow when using larger amounts of data. For these reasons we do not recommend using individual utterances as the data units to split, but rather some grouping of them that will limit the number of nodes in the graph to a few thousand.

In practice we can run our splitting algorithms on Librispeech sized corpora in 1-3 hours, where we used chapters as the data unit for splitting. That example is included in test/fixtures/disconnected_fraction_eg.tsv and has about 6,000 entries. On smaller corpora, an example of which we have also included in test/fixtures/connected_fraction_constraints_eg.tsv with 450 entires, the splitting is very fast. Again much of this depends on the underlying structure of the graph, and the number of factors and constraints used. Thus far the splitting algorithms have scaled sufficiently well for our use-cases, but we suspect that applying similar technique to much larger dataset, which is a goal we ultimately have, may require some refactoring of the graph-based algorithms.
