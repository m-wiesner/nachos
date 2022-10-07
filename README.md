# NACHOS -- Nearly Automatic Creation of Held Out Splits

This toolkit provide several methods for creating heldout splits using a 
file containing metadata about the units over which we are splitting. These
metadata may include features such as the speaker(s) present in a recording,
the gender of the speaker, the duration of the recording, the prompt spoken, 
the room in which it was spoken etc...

## What you need to do

Create a file representing your corpus that has a column for the id of each
element of the corpus (i.e., the one you are trying to split), and then a
column for each feature that is relevant for creating the split. This could be
prompt, speaker ID, etc..

We support multiple splitting methods, (splitters), which appear to be better
suited for different tasks.

SpectralClusteringSplitter -- when the data are such that each item is similar
to all other, i.e., it forms a complete graph, then there are no heldout sets
which can be formed from this data that don't overlap in some feature of
interest. In this case, the problem of finding a heldout set is relaxed and
we instead try to find two sets of similar size that have minimal overlap. In
practive we have found that the one of the splits tends to be somewhat larger
and could easily serve as the training set.


RandomFeatureSplitter -- when the data are such that each item is similar to
other items, which are in term similar to other, until every item in the data
has been seen, then this creates a connected graph, i.e., having a single
component. When we want to create a held-out set using these data, especially
if the training set is somewhat smaller and the heldout sets are larger, the
RandomFeatureSplitter is a good choice.


MinNodeCutSplitter -- when the data are as in the RandomFeatureSplitter case, 
but we want to create the largest possible cluster, this method may be better.  

The list of splitters we support can be found in splitters/

Below are example of how to split data sets.

MinNodeCutSplitter
```
python run.py \
  --log log \
  --simfuns set_intersect set_intersect \
  --max-iter 40 \
  --metrics overlap overlap \
  --feature-name spkr room \
  --splitter MinNodeCutSplitter \
  --seed 0 \
  --train-ratio 0.8 \
  --heldout-ratio 0.1 \
  mx6_features_multilbl mx6_splits
```

SpectralClusteringSplitter
```
python run.py \
  --num-splits 3 \
  --log log \
  --simfuns set_intersect set_intersect \
  --metrics overlap overlap \
  --feature-name spkr room \
  --splitter SpectralClusteringSplitter \
  mx6_features_multilbl mx6_splits
```

RandomFeatureSplitter
```
python run.py \
  --log log \
  --simfuns set_intersect set_intersect \
  --max-iter 40 \
  --metrics overlap overlap \
  --feature-name spkr room \
  --splitter RandomFeatureSplitter \
  --seed 0 \
  --train-ratio 0.8 \
  --heldout-ratio 0.1 \
  mx6_features_multilbl mx6_splits
```

