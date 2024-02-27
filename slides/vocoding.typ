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
  - Spectrogram generating techniques normally generate amplitude not phase
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
  - Testing with multiple realistic inputs is the most valuable
  - Aside from that look to build a model and use unit testing to test your understanding of it
]

#slide[
  == Notes on Implementation

  - This was done as a port from librosa as a comparative benchmark
  - Wanted to compare our new neural vocoder versus well-understood non-neural implementation
  - Never seen production, and while it's tested it's _less tested_
]
