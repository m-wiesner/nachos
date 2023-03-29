# NACHOS -- Nearly Automatic Creation of Held Out Splits

This toolkit provide several methods for creating heldout splits using a 
file containing metadata about the units over which we are splitting. These
metadata may include features such as the speaker(s) present in a recording,
the gender of the speaker, the duration of the recording, the prompt spoken, 
the room in which it was spoken etc...

## Installation
```
git clone https://github.com/m-wiesner/nachos.git
cd nachos 
pip install -r requirements.txt
pip install -e . 
```
## What you need to do

Create a file representing your corpus that has a column for the id of each
element of the corpus (i.e., the one you are trying to split), and then a
column for each feature that is relevant for creating the split. This could be
prompt, speaker ID, etc..

We will support multiple splitting methods, (splitters), which appear to be better
suited for different tasks. For now we have implemented:
   a random feature splitting method (random)
   minimum node cut splitter (min_node_cut)
   variable neighborhood search splitter (vns)

## Running

Set the values in config.yaml to the desired values and then

```
python run.py
```
