# NACHOS -- Nearly Automatic Creation of Held Out Splits

This toolkit provide several methods for creating heldout splits using a 
file containing metadata about the units over which we are splitting. These
metadata may include features such as the speaker(s) present in a recording,
the gender of the speaker, the duration of the recording, the prompt spoken, 
the room in which it was spoken etc...

We support one method called the SpectralClusteringSplitter, which creates an
affinity matrix for the provided feature file and creates the specified number
of clusters from the affinity matrix. This corresponds to a normalized min-cut
problem. For the SpectralClusteringSplitter, we support multilabel features
if the overlap similarity function if used.

The list of splitters we hope to support can be found in splitters/

Below is an example of how to run the method.

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

