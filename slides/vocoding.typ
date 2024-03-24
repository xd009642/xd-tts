#import "@preview/polylux:0.3.1": *
#import "theme.typ": *;

#show: talk-theme

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#focus-slide[
  == The Vocoder
]

#slide[
  == Turning It Into Sound

  - In the frequency domain we have magnitude and phase information
  - Magnitude is easier to learn 
  - Spectrogram generating techniques normally generate magnitude not phase
  - We can get the parameters for vocoding from the Tacotron2 repo
]

#slide[
    #align(center)[#image("images/mag_vs_phase.svg")]
]


#slide[
  == Griffin-Lim Basics

  - Convert from mel spectrogram to linear spectrogram
  - Create random phase spectrum 
  - Convert to audio and then back to spectrogram
  - Restore the magnitude because _maths_.
  - Repeat until stopping condition reached
]

#slide[
  == How Do We Test It?

  - We have a reference golden implementation
  - Gather outputs from it and do a comparison
    - Comparing matrices of floats is a bit painful (lose developer UX)
  - Testing with realistic inputs is the most valuable
  - Aside from that learn and use unit testing to test your understanding
]

#slide[
  == Notes on Implementation

  - This was done as a port from librosa 
  - Wanted to compare with a well-understood analytic approach 
  - Never seen production, and while it's tested it's _less tested_
  
  #align(right + bottom)[#image("images/corro.svg", width: 20%)]
]

#slide[
  == How Does it Sound?

  - Griffin-Lim doesn't have a model of how human speech sounds
  - It just tries to do something simple and quick
  - As a result there's some artefacts

  #align(right + bottom)[#image("images/ferris-listen.png", width: 20%)]
]
