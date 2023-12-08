"""Convert model to onnx

echo "One sentence. \nAnother sentence. | python code/convert.py"

Run from the project root.
Does not handle numbers - write everything in words.

Add this to the speedyspeech repo and run it to export the model. You'll likely need to 
upgrade the torch in their requirements.txt and add onnx to it in order to work.
"""
import argparse, sys, os, time
import torch
from librosa.output import write_wav

from speedyspeech import SpeedySpeech
from melgan.model.generator import Generator
from melgan.utils.hparams import HParam
from hparam import HPStft, HPText
from utils.text import TextProcessor
from functional import mask


speedyspeech_checkpoint = 'checkpoints/speedyspeech.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
audio_folder = 'synthesized_audio'

print('Loading model checkpoints')
m = SpeedySpeech(
    device=device
).load(speedyspeech_checkpoint, map_location=device)
m.eval()


print('Processing text')
txt_processor = TextProcessor(HPText.graphemes, phonemize=HPText.use_phonemes)
text = [t.strip() for t in sys.stdin.readlines()]

phonemes, plen = txt_processor(text)
# append more zeros - avoid cutoff at the end of the largest sequence
phonemes = torch.cat((phonemes, torch.zeros(len(phonemes), 5).long() ), dim=-1)
phonemes = phonemes.to(device)
plen = torch.tensor(plen)

print(phonemes)
print(plen)

print('Synthesizing')
# generate spectrograms
with torch.no_grad():
    spec, durations = m((phonemes, plen))

x = [
         phonemes,
         plen
]

print(x)

# Export the model
torch.onnx.export(m,                                # model being run
                  x,                                # model input (or a tuple for multiple inputs)
                  "model.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,               # store the trained parameter weights inside the model file
                  verbose=True,
                  opset_version=12  ,                 # the ONNX version to export the model to
                  do_constant_folding=True,         # whether to execute constant folding for optimization
                  input_names = ['phonemes', 'plen'],    # the model's input names
                  output_names = ['spec', 'durations'],  # the model's output names
                  #enable_onnx_checker = True,
                  dynamic_axes={
                                'phonemes' : {0: 'batch_size', 1: 'sequence'},
                                'plen' : {0: 'batch_size'},
                                'spec' : {0: 'batch_size', 1: 'time'},
                                'durations' : {0: 'batch_size', 1: 'sequence'}
                                }
                  )

"""Check the vocoder ONNX file.

"""

import onnx

# Load the ONNX model
model = onnx.load("model.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

print(model.graph.input)
print(model.graph.output)
