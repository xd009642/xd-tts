#import "@preview/polylux:0.3.1": *
#import themes.simple: *

#import "@preview/cades:0.3.0": qr-code

#set page(paper: "presentation-16-9")
#set text(size: 25pt)


#focus-slide[
    #qr-code("https://github.com/emotechlab/griffin-lim", width: 10cm)
    #link("https://github.com/emotechlab/griffin-lim")
]
