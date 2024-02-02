#import "@preview/polylux:0.3.1": *
#import "@preview/diagraph:0.2.1": *
#import themes.simple: *

#import "@preview/cades:0.3.0": qr-code

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#focus-slide[
  == Text Normalisation 
]

#slide[
  == Text Normalisation

  - Convert text from written form to spoken form
  - Traditionally systems were rule based but statistics and neural networks have offered improvements
  - A lot of people go for hybrid systems to enable tailoring normalisation by domain
  - For our system we're going to do a fairly traditional rule based approach
]

#slide[
  == Challenges

  - For the rules we need to identify to some level what each token is
  - Is a number a year, currency, date, phone number, ordinal or cardinal?
  - Is a sequence of capital letters an initialism or shouting?
  - *How can we let users guide pronunciation?*
]

#focus-slide[
  Luckily there's a W3C standard for that
  
  _And people actually use it!_
]

#slide[
  == SSML

  - Speech Synthesis Markup Language an XML spec to guide a speech synthesiser
  - Can use XML tags to give instructions to a TTS engine
  - Best to build in support from day 1 - it can drive normalisation
]

#slide[
  == Example SSML

  ```xml
<speak>
    I have <break time="3s"/> 
    <say-as interpret-as='cardinal'>1</say-as> 
    <phoneme alphabet='ipa' ph='ˈpi.kæn'>pecan</phoneme>
</speak>
  ```
]

#focus-slide[
    #qr-code("https://github.com/emotechlab/ssml-parser", width: 10cm)
    #link("https://github.com/emotechlab/ssml-parser")
]

#slide[
  == Notable Rust Pattern!

  ```rust 
  pub enum Element {
     Break,
  }
  pub enum ParsedElement {
     Break(BreakAttribute),
  }
  // The first one feels more normal to newcomers
  parsed_element.tag() == Element::Break;
  matches!(parsed_element, ParsedElement::Break(_));
  // impl ParsedElement::tag is left as an exercise to reader
  ```

]

#slide[
  == Key Crates
  
  Implementation

  - Num2Words
  - Regex
  - Deunicode
  - Unicode-segmentation
  - Quick-xml
]

#slide[
  == Back to Text Normalisation!

  - General approach is to turn output into a list of chunks
  - These are either: text, phonemes, tts state changes
  - Then we can stop normaliser normalising things SSML has defined
  - And store SSML state changes
  - For text we split by spaces, grab punctuation then try to normalise each word
  - Keeping it simple!
]

#slide[
  == So Pronunciation?

  - After normalisation often we turn words to phonemes
  - For simplicity here we use a dictionary lookup approach
  - For unseen words G2P (Grapheme to Phoneme) models are used.
  - Older models needed to align phonemes to graphemes for a one-to-one mapping.
 
    #raw-render(
      ```dot
        digraph G {
           a
           c1[label="c"]
           c2[label="c"]
           u
           s
           e

           AH0
           K
           Y
           UW1
           Z

           a->AH0
           c1->K
           c2->K
           u->Y
           u->UW1
           s->Z
        }

      ```
    )
]
