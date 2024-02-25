#import "@preview/polylux:0.3.1": *

// #set dark_background(rgb("#1a1a25"))

#let title_text(content) = {
    set text(fill: gradient.linear(rgb("#42d2e4"), rgb("#38ce98")))
    box(content)
}

#let talk-theme(
  aspect-ratio: "16-9",
  footer: [],
  background: white,
  foreground: rgb("#1a1a25"),
  body
) = {
  set page(
    paper: "presentation-" + aspect-ratio,
    margin: 2em,
    header: none,
    footer: none,
    fill: background,
  )
  set text(font: "Roboto", fill: foreground, size: 25pt)
  show footnote.entry: set text(size: .6em)
  show heading.where(level: 2): set block(below: 2em)
  set outline(target: heading.where(level: 1), title: none, fill: none)
  show outline.entry: it => it.body
  show outline: it => block(inset: (x: 1em), it)

  body
}

#let title-slide(title: [], author: []) = {
  set page(
    paper: "presentation-16-9",
    margin: 2em,
    header: none,
    footer: none,
    fill: rgb("#1a1a25")
  )
  set text(font: "Roboto", size: 35pt)
  polylux-slide({
    set align(center + horizon)
    title_text(strong(title))
    parbreak()
    text(size: .7em, text(fill: white, author))
  })
}


#let slide(title: [], body) = {
  polylux-slide({
    strong(title)
    set align(top)
    body
  })
}

#let focus-slide(body) = {
  set page(
    paper: "presentation-16-9",
    margin: 2em,
    header: none,
    footer: none,
    fill: rgb("#1a1a25")
  )
  set text(size: 35pt, fill: white)
  polylux-slide(align(center + horizon, title_text(body)))
}
