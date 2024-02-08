#import "@preview/polylux:0.3.1": *
#import "theme.typ": *;

#show: talk-theme

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#slide[
  == The Fundamentals

  - We can break a word into a few different units:
    - Letters (graphemes)
    - Syllables 
    - Phonemes (smallest unit of speech)
] 

#slide[
  == Phonemes

  - The words crab and cram are a single syllable, but they sound different
  - IPA is a common phoneme alphabet but speech applications often use ARPABET
  - We're using ARPABET because the CMU dictionary uses it 
  - CMU dict is an open source pronunciation dictionary for English.
  - Phonemes may also carry information about pitch, stress and rhythm if it is phonologically relevant
    - Pitch is very useful in tonal languages
]

#slide[
  == Advantage of Phonemes

  - Traditionally we want to move a transcript into a smaller unit and silences
  - Phonemes are a good element for this
  - End to end deep learning systems sometimes won't use them
  - But this lacks control. How do we ensure every word is pronounced correctly!?
]

#slide[
  == Is that it?

  - No! As well as making the sounds correctly we want to model prosody
  - Speech should have a natural intonation and rhythm
  - This differs language to language.
  - English is stress-timed, Mandarin syllable timed and Japanese mora timed
  - Modern systems rely on a HMM or DNN learning the prosody. Given enough examples we can predict it
]
