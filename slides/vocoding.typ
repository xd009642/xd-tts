#import "@preview/polylux:0.3.1": *
#import "theme.typ": *;

#show: talk-theme

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#focus-slide[
  == The Vocoder
]

#qr-slide(url: "https://github.com/emotechlab/griffin-lim")

#slide[
  == Turning it into Sound

  - In the frequency domain we have magnitude and phase information
  - How tall a frequency is and the delay applied to the signal
  - Frequency amplitudes contain more information than phase, phase looks random in comparison
  - Because of this spectrogram generating techniques normally generate amplitude not phase
]

#slide[
    #align(center)[#image("images/mag_vs_phase.svg")]
]


#slide[
  == Griffin Lim Basics

  - Create random phase spectrum 
  - Convert to audio and then back to spectrogram
  - Restore the magnitude because it should be invariant
  - Repeat until stopping condition reached
]

#slide[
    == How do we test it?

    - We have a reference golden implementation
    - Gather outputs from it and do a comparison
    - Testing with multiple realistic inputs is the most valuable
    - Aside from that look to build a model and use unit testing to test your understanding of it
]


