#import "@preview/polylux:0.3.1": *
#import "@preview/diagraph:0.2.1": *
#import "theme.typ": *;

#show: talk-theme

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#title-slide(
  title: [Creating a Text-To-Speech System in Rust],
  author: [Daniel McKenna (xd009642)]
)

#slide[
  == Introduction

  - Rust programmer at Emotech an AI startup primarily using Rust
  - Primarily working in speech technologies and related areas
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
  - Some researchers create C++ based systems (becoming less popular recently)
]

#slide[
  == What's hard about Text to Speed?

  - Language is hard
    - Unknown words - proper nouns, made up words
    - Homographs: lead, bass, bow
    - Code-switching
  - Speech is hard it has to sound natural - rhythm, tone, stress
  - Naturalness often conflicts with intelligibility
  - Users want it controllable
]

#include "historic_systems.typ"

#slide[
  == Our System
  #align(center)[
      #raw-render(
        ```dot
          digraph G {
            text[label="Text Normalisation", shape=Box]
            melgen[label="Spectrogram Generation", shape=Box]
            vocoding[label="Vocoder", shape=Box]
            text->melgen->vocoding
          }
        ```,
        height: 70%
      )
    ]
]

#include "what_is_speech.typ"

#include "text_processing.typ"

#include "melgen.typ"

#include "vocoding.typ"

#slide[
    == The Links!
    
    - https://github.com/xd009642/xd-tts
    - https://github.com/emotechlab/ssml-parser
    - https://github.com/emotechlab/griffin-lim
]

#focus-slide[
    == Any Questions?
]
