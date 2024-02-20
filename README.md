# Rustnation 2024 TTS in Rust

This is the project repository for my 2024 Rustnation talk _Creating a
Text-To-Speech System in Rust_. Here you can find the main TTS engine, slides
and a collection of useful scripts and resources. Check out each modules
documentation for more content I didn't have time to discuss in the talk!

For the other projects mentioned look no further:

* [ssml-parser](https://github.com/emotechlab/ssml-parser)
* [griffin-lim](https://github.com/emotechlab/griffin-lim)

## Getting Started

This repo uses [git-lfs](https://git-lfs.com/) to store the neural networks,
make sure this is setup before
cloning or things may not work as expected.

I'm also using the dynamic library feature for ORT. You will need to download
the correct ORT version for your system
[here](https://github.com/microsoft/onnxruntime/releases/tag/v1.17.0) 
and set the `ORT_DYLIB_PATH` env var to the path to `libonnxruntime.so`.
Alternatively, if the ORT project downloads the correct version for your system
you can manually remove the feature.

There are two binaries in the project, one to prepare/analyse training data and
another to run the TTS

## Other folders

### Slides

The presentation slides! These are done using [typst](https://typst.app/).

### Scripts

Scripts here are mainly for some dataset cleaning, and plotting scripts to
generate images for the slides. There's also a folder inside called
speedyspeech for an old and largely abandoned part of the project.

### Resources

In the resources folder I've added a custom dictionary, this includes
tokens in the LJ Speech corpus which aren't present in CMU Dict. For this I've
used the `data_cleaning.py` script in scripts and the gruut grapheme-to-phoneme
(g2p) models. If I'd had time to do my own g2p this would have also been pure
Rust.

For `data_cleaning.py` you will need to download the librispeech lexicon
[here](https://openslr.trmal.net/resources/11/librispeech-lexicon.txt)


## Old Bits and Bobs

### SpeedySpeech

There is also disabled support for loading a pre-trained speedy speech model
where we load it via candle/torch/tract. Unfortunately, due to ONNX support
outside of ORT this ended up being abandoned. But the code should work for other
ONNX models, or JIT traced torch models which work better for those
dependencies. 

Unfortunately, I can't pretend any of it is useful, but for someone considering
using any of those crates these modules can be a pointer on how to start using
them. I've also added a vast array of doc comments explaining some of the
conversion process and difficulties I faced.

## References

* [Tacotron2 Paper](https://arxiv.org/abs/1712.05884)
* [netron.app - an ONNX viewer](https://netron.app/)
* [ssml-parser](https://github.com/emotechlab/ssml-parser)
* [griffin-lim](https://github.com/emotechlab/griffin-lim)
* [speech.zone - great for learning about speech AI](https://speech.zone/)
* [ONNX graph surgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)
* [typst](https://typst.app/)
