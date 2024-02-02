#import "@preview/polylux:0.3.1": *
#import themes.simple: *

#import "@preview/cades:0.3.0": qr-code

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#focus-slide[
  == The Vocoder
]

#focus-slide[
    #qr-code("https://github.com/emotechlab/griffin-lim", width: 10cm)
    #link("https://github.com/emotechlab/griffin-lim")
]

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

  - Transform Mel spectrogram to a linear spectrogram
  - Apply some gain to the spectrogram as signal ranges may be a bit low
  - Create random phase spectrum 
  - Apply inverse stft to the phase and magnitude spectrum
  - Reapply stft
  - Compare magnitude spectrum to our reference one to get an error
  - Apply a delta and repeat until iterations reached or convergence
]
