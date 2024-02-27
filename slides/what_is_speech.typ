#import "@preview/polylux:0.3.1": *
#import "theme.typ": *;

#show: talk-theme

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#focus-slide[
    == What is Speech?
]

#slide[
  == The Fundamentals

  - We can break a word into a few different units:
    - Letters (graphemes)
    - Syllables 
    - Phonemes - language specific sound that disambiguates words. Can include things like pitch
    - Phones - smallest unit of speech
] 

#slide[
  == Phonemes

  - The words crab and cram are a single syllable, but they sound different
  - IPA is a common phoneme alphabet but some speech applications use ARPABET
  - CMU dict is an open source ARPABET pronunciation dictionary 
]

#slide[
  == Advantage of Phonemes

  - Traditionally we turn text into a smaller units 
  - Phonemes are a good element for this
  - End to end deep learning systems sometimes won't use them
  - But this lacks control. How do we ensure every word is pronounced correctly!?
]

#slide[
  == Is that it?

  - No! As well as making the sounds correctly we want to model prosody
  - Speech should have a natural intonation and rhythm
  - This differs language to language.
  - Languages can be stress, syllable (or mora) timed
]
