use deunicode::deunicode;
use once_cell::sync::OnceCell;
use regex::Regex;
use ssml_parser::{elements::*, parser::SsmlParserBuilder, ParserEvent};
use std::time::Duration;

#[derive(Clone, Debug)]
enum NormaliserChunk {
    Text(String),
    Break(Duration),
    Pronunciation(String),
}

#[derive(Clone, Debug, Default)]
pub struct NormalisedText {
    chunks: Vec<NormaliserChunk>,
}

impl NormalisedText {
    pub fn text(&self) -> String {
        self.chunks.iter().fold(String::new(), |mut acc, x| {
            if let NormaliserChunk::Text(t) = x {
                acc.push_str(t.as_str())
            }
            acc
        })
    }
}

pub fn dict_normalise(x: &str) -> String {
    // This regex is just for duplicate pronunciations in CMU dict
    static VERSION_REGEX: OnceCell<Regex> = OnceCell::new();
    let version_regex = VERSION_REGEX.get_or_init(|| Regex::new(r#"\(\d+\)$"#).unwrap());

    normalise_text(&version_regex.replace_all(x, ""))
}

pub fn normalise_ssml(x: &str) -> anyhow::Result<NormalisedText> {
    let parser = SsmlParserBuilder::default().expand_sub(true).build()?;

    let mut res = NormalisedText::default();
    let mut stack = vec![];
    let mut pending_text = String::new();
    let mut push_text = true;
    for event in parser.parse(x)?.event_iter() {
        match event {
            ParserEvent::Text(t) => {
                if push_text {
                    pending_text.push_str(&t);
                } else if let Some(tag) = stack.last() {
                    // We should look at the stack to see if there's something we're meant to be
                    // doing
                    match tag {
                        ParsedElement::SayAs(sa) => match sa.interpret_as.as_str() {
                            "ordinal" => {}
                            "cardinal" => {}
                            "characters" => {}
                            s => eprintln!("Unsupported say-as: {}", s),
                        },
                        ParsedElement::Phoneme(ph) => {}
                        _ => unreachable!(),
                    }
                } else {
                    println!("I don't know what to do with myself");
                }
            }
            ParserEvent::Open(open) => {
                match &open {
                    ParsedElement::SayAs(sa) => {
                        push_text = false;
                    }
                    ParsedElement::Phoneme(ph) => {
                        push_text = false;
                    }
                    ParsedElement::Emphasis(em) => {}
                    ParsedElement::Prosody(pr) => {}
                    e => {
                        eprintln!("Unhandled open tag: {:?}", e);
                    }
                }
                stack.push(open);
            }
            ParserEvent::Close(_close) => {
                if let Some(_end) = stack.pop() {
                    // Assume we only go one deep
                    push_text = true;
                } else {
                    unreachable!();
                }
            }
            ParserEvent::Empty(tag) => match &tag {
                ParsedElement::Break(ba) => {}
                e => {
                    eprintln!("Unhandled tag: {:?}", tag);
                }
            },
        }
    }

    todo!()
}

pub fn normalise_text(x: &str) -> String {
    let mut s = deunicode(x);
    s.retain(valid_char);
    s.make_ascii_uppercase();

    s
}

fn valid_char(x: char) -> bool {
    !r#"!"£$%^&*()-_=+[{]};:'@#~,<.>/?|\`¬"#.contains(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_removal() {
        assert_eq!(normalise_text("BATH(2)"), "BATH");
        assert_eq!(normalise_text("HELLO!(45)"), "HELLO");
        assert_eq!(normalise_text("(3)d"), "3D");
    }

    #[test]
    fn normalise_ssml() {
        let text = r#"<speak>
        <say-as interpret-as="characters">SSML</say-as> 
        </speak>"#;
        let expected = "S S M L";

        assert_eq!(normalise_ssml(text).unwrap().text(), expected);

        let text = r#"<speak>
        <say-as interpret-as="cardinal">10</say-as> 
        </speak>"#;
        let expected = "TEN";

        assert_eq!(normalise_ssml(text).unwrap().text(), expected);

        let text = r#"<speak>
        <say-as interpret-as="ordinal">10</say-as>
        </speak>"#;
        let expected = "TENTH";

        assert_eq!(normalise_ssml(text).unwrap().text(), expected);

        let text = r#"<speak>
        <say-as interpret-as="characters">10</say-as>.
        </speak>"#;
        let expected = "ONE ZERO";

        assert_eq!(normalise_ssml(text).unwrap().text(), expected);

        let text = r#"<speak>
        <sub alias="World Wide Web Consortium">W3C</sub>#";
        </speak>"#;
        let expected = "WORLD WIDE WEB CONSORTIUM";

        assert_eq!(normalise_ssml(text).unwrap().text(), expected);
    }
}
