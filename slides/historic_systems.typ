#import "@preview/polylux:0.3.1": *
#import "theme.typ": *;

#show: talk-theme

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
  - By modelling how they change we can combine and make a sound
  - Good intelligibility and runtime but sounds robotic
  - Very low-level modelling of speech so hard to develop
]

#slide[
  == Concatenative Synthesis
 
  - We have a database of audio samples for "units"
  - Sub-word units e.g. syllables, phonemes, diphones
  - We concatenate them to make audio
  - Sounds natural except where the samples join there may be glitches
]

#slide[
  == HMM Based Synthesis

  - A statistical model of speech based on Hidden Markov Models
  - Implementations typically use HTK - a C library
  - Was state-of-the-art pre-deep learning.
  - Duration modelling is tricky!
]

#slide[
  == Deep Learning

  - Uses neural networks and a lot more data
  - Typically one of 2 flavours:
    - Generates audio (end-to-end model)
    - Generates spectrogram then a vocoder (neural or otherwise) generates audio
]
