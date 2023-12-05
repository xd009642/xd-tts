# retro-tts

1. Turn a sentence into lingustic features i.e. phonemes and duration
2. Process lingustic features to vocoder features i.e. cepstra, spectrogram, fundamental frequency
3. Generate audio

## Text to Phonemes

## Phonemes to Vocoder Features

* Consider triphones for past and future context

### Prosody

Prosody refers to certain properties of speech such as changes in pitch,
loudness and syllable length. It can also include stresses, and speech rate.
Prosodic events are time-aligned with syllables or groups of syllables instead
of segments (sounds, phonemes) so are referred to as suprasegmental phenomena -
or suprasegmentals.

It can be represented as follows:

* Acoustic level - the representation of it in sound waves. Frequency, duraion, amplitude.
* Perceptual level - how a person hears it 
* Lingustic level - the functional analysis of prosody by a lingust - more descriptive of prosody

## SpeedySpeech

There is also support for loading a pre-trained speedy speech model where we load it via candle. To
do this download the latest SpeedySpeech mode as so:

```
wget -O models/speedyspeech.pth \
    https://github.com/janvainer/speedyspeech/releases/download/v0.2/speedyspeech.pth 
```

Then there's a script to convert to onnx but onnx wasn't able to be loaded. Then there's issues because
speedyspeech torch is so old and it needs JIT tracing to load in Rust which involves inference and
then tracing the ops. So some dynamic sized tensors may be removed internal to the model in a way we
don't want.

So I converted from onnx to tensorflow by:

```
pip install onnx-tf
onnx-tf convert -t tf -i models/speedyspeech.onnx -o models/speedyspeech.pb
```

This feels like a lot of work just to export a model but yeah... Tensorflow and Torch projects can
both be painful if not designed with exporting out of the python scripts in mind. And it also didn't work because
of the hit-and-miss support for ONNX in the ecosystem. ONNX is not yet the solution to all our woes.

## Vocoding

## References

* [An Introduction to HMM-Based Speech Synthesis - Junichi Yamagishi](https://wiki.inf.ed.ac.uk/pub/CSTR/TrajectoryModelling/HTS-Introduction.pdf)
* [CMU Dict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
* [Speech and Language Processing TTS (Cambridge University slides)](https://mi.eng.cam.ac.uk/~pcw/local/4F11/4F11_2014_lect14.pdf)
* [Software for a cascade/parallel formant synthesizer - Dennis H. Klatt](https://www.fon.hum.uva.nl/david/ma_ssp/doc/Klatt-1980-JAS000971.pdf)
* [A beginners’ guide to statistical parametric speech synthesis - Simon King](https://www.cs.brandeis.edu/~cs136a/CS136a_docs/king_hmm_tutorial.pdf)
* [Speech representation and transformation using adaptive interpolation of weighted spectrum: vocoder revisited - Hideki Kawahara](https://www2.spsc.tugraz.at/people/franklyn/ICASSP97/pdf/scan/ic971303.pdf)
* [An Introduction to Text-to-Speech Synthesis - Thierry Dutoit](https://books.google.co.uk/books?id=sihrCQAAQBAJ)
* [netron](https://netron.app/)
* [Standford lecture on neural TTS](https://web.stanford.edu/class/cs224s/lectures/224s.22.lec16.pdf)
