#import "@preview/polylux:0.3.1": *
#import "@preview/diagraph:0.2.1": *
#import "theme.typ": *;

#show: talk-theme

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#focus-slide[
  == Text Normalisation 
]

#slide[
  == Text Normalisation

  - Convert text from written form to spoken form
  - Traditionally systems were rule based but statistics and neural networks have offered improvements
  - A lot of people go for hybrid systems for customisation 
  - For our system we're going to do a fairly traditional rule based approach
  - unicode segmentation and deunicode crates are exceptionally useful and easy to use!
]

#slide[
  == Challenges

  - For the rules we need to identify to some level what each token is
  - For example there's a lot of ways to read out numbers like 1971
  - Is a sequence of capital letters an initialism or shouting?
  - *Could we simplify normalisation and can we let users guide pronunciation?*
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
  == But We Still Have to Normalise

  - General approach is to turn output into a list of chunks
  - These are either: text, phonemes, tts state changes
  - Then we can stop normaliser normalising things SSML has defined
  - And store SSML state changes
  - For text we split by spaces, grab punctuation then try to normalise each word
  - Keeping it simple!
]

#slide[
  == The Final Step

  - After normalisation often we turn words to phonemes
  - For simplicity here we use a dictionary lookup approach
  - For unseen words G2P (Grapheme to Phoneme) models are used.
  - Older models needed to align phonemes to graphemes for a one-to-one mapping.
  - Having incorrect mappings to make your model work isn't great!
 
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
