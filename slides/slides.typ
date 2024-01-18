#import "@preview/polylux:0.3.1": *

#import themes.simple: *

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#title-slide[
  = Creating a Text-To-Speech System in Rust
  #v(2em)

  Daniel McKenna (xd009642)
]

#slide[
  == Introduction

  - Rust programmer at Emotech an AI startup primarily using Rust
  - Primarily working in speech technologies and related areasA
  - xd009642 online
  - May know me from cargo-tarpaulin
]

#slide[
  == And this Talk?

  - Introduce TTS systems and the challenges
  - Cover all the stages of a pipeline
  - Demonstrating it all with an open source TTS engine made for this talk!
]

#slide[
  == Why Rust?

  - Sometimes these AI systems need to be "real time"
  - Also handle load from API users 
  - Python breaks down pretty quickly in this scenario
]

#slide[
  == Better than C++

  - Because of real time requirements and research into streaming systems a lot of speech things use C++
  - Kaldi and HTK are two C++ libraries still used in speech
  - These are becoming less prevalent given success of recent E2E models
  - Those models are typically non-streaming and not suitable for edge or CPU deployments
]

#include "what_is_speech.typ"

#slide[
  == What's hard about Text to Speed?

  - Language is hard
    - Unknown words - proper nouns, made up words
    - Homographs: lead, bass, bow
  - Speech is hard it has to sound natural - rhythm, tone, stress
  - Also has to be intelligible
  - Users want it controllable
]

#include "historic_systems.typ"

#slide[
  == Our System

]

#include "text_processing.typ"

#include "melgen.typ"

#include "vocoding.typ"
