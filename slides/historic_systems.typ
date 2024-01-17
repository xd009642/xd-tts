#import "@preview/polylux:0.3.1": *
#import themes.simple: *

#set page(paper: "presentation-16-9")
#set text(size: 25pt)


#slide[
  == How have we done TTS in the past?

  - Formant Synthesis
  - Concatenative Synthesis
  - HMM Based Synthesis
  - Deep learning 
  - And of course hybrid systems of the above
]

#slide[
  == Formant Synthesis

  - A formant is a resonance of the vocal tract 
  - Adding them together creates sounds
  - By modelling how they change during phonemes we can add them together and make the sound
  - Good intelligibility and runtime but sounds robotic
  - Very low level modelling of speech so hard to develop
]

#slide[
  == Concatenative Synthesis
 
  - We have a database of audio samples for "units"
  - These are something like syllables, phonemes, diphones
  - We concatenate them to make audio
  - Sounds natural except where the samples join there may be glitches
]

#slide[
  == HMM Based Synthesis

  - A statistical model of speech based on Hidden Markov Models
  - Typically uses frequency spectrum, fundamental frequency and duration in the model
  - Then uses statistical maximisation to generate audio from this
  - Implementations typically use HTK - a C library. Some complexities most HMM libraries don't tackle
  - Was state of the art pre-deep learning. Duration modelling is an issue though!
]

#slide[
  == Deep Learning

  - Uses neural networks and a lot more data
  - Typically one of 2 flavours:
    - Generates audio (end to end model)
    - Generates spectrogram then a vocoder (neural or otherwise) generates audio
]
