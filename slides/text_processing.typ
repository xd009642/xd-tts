#import "@preview/polylux:0.3.1": *
#import themes.simple: *

#import "@preview/cades:0.3.0": qr-code

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#focus-slide[
  _Luckily there's a W3C standard for that_
  
  And people actually use it!
]

#focus-slide[
    #qr-code("https://github.com/emotechlab/ssml-parser", width: 10cm)
    #link("https://github.com/emotechlab/ssml-parser")
]
