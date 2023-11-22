use crate::phonemes::Unit as TtsUnit;
use crate::phonemes::*;
use crate::training::CmuDictionary;
use deunicode::deunicode;
use num2words::Num2Words;
use once_cell::sync::OnceCell;
use regex::Regex;
use ssml_parser::{elements::*, parser::SsmlParserBuilder, ParserEvent};
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, error, info, warn};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Clone, Debug)]
pub enum NormaliserChunk {
    Text(String),
    Break(Duration),
    Pronunciation(Vec<TtsUnit>),
    Punct(Punctuation),
}

#[derive(Clone, Debug, Default)]
pub struct NormalisedText {
    chunks: Vec<NormaliserChunk>,
}

impl NormalisedText {
    pub fn words_to_pronunciation(&mut self, dict: &CmuDictionary) {
        for x in self
            .chunks
            .iter_mut()
            .filter(|x| matches!(x, NormaliserChunk::Text(_)))
        {
            let s = if let NormaliserChunk::Text(s) = x {
                s.clone()
            } else {
                unreachable!()
            };
            let mut units = vec![];
            for word in s.split_ascii_whitespace() {
                if let Some(pronunciation) = dict.get_pronunciations(word) {
                    assert!(!pronunciation.is_empty());
                    info!("{} is pronounced: {:?}", word, pronunciation);
                    units.extend(pronunciation[0].iter().map(|x| TtsUnit::Phone(*x)));
                    units.push(TtsUnit::Space);
                } else {
                    warn!("Unsupported word: '{}'", word);
                }
            }
            *x = NormaliserChunk::Pronunciation(units);
        }
    }

    /// Draining iterator, takes all the chunks out
    pub fn drain_all(&mut self) -> impl Iterator<Item = NormaliserChunk> + '_ {
        self.chunks.drain(..)
    }

    /// Ignores breaks, only looks at punctuation and text. If pronunciation present will fail
    pub fn to_string(&self) -> anyhow::Result<String> {
        let mut res = String::new();
        for chunk in &self.chunks {
            match chunk {
                NormaliserChunk::Text(s) => res.push_str(&s),
                NormaliserChunk::Punct(p) => res.push_str(&p.to_string()),
                NormaliserChunk::Pronunciation(_) => {
                    anyhow::bail!("Can't turn pronunciation chunk into text")
                }
                NormaliserChunk::Break(_) => {}
            }
        }
        Ok(res)
    }
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

pub fn normalise(x: &str) -> anyhow::Result<NormalisedText> {
    if x.contains("<speak") {
        normalise_ssml(x)
    } else {
        Ok(NormalisedText {
            chunks: vec![NormaliserChunk::Text(normalise_text(x))],
        })
    }
}

pub fn dict_normalise(x: &str) -> String {
    // This regex is just for duplicate pronunciations in CMU dict
    static VERSION_REGEX: OnceCell<Regex> = OnceCell::new();
    let version_regex = VERSION_REGEX.get_or_init(|| Regex::new(r#"\(\d+\)$"#).unwrap());

    normalise_text(&version_regex.replace_all(x, ""))
}

fn handle_say_as(say_as: &SayAsAttributes, text: &str) -> anyhow::Result<NormaliserChunk> {
    match say_as.interpret_as.as_str() {
        "ordinal" => {
            let num = text.trim().parse::<i64>()?;
            let text = Num2Words::new(num)
                .ordinal()
                .to_words()
                .map_err(|e| anyhow::anyhow!(e))?
                .replace("-", " ")
                .to_ascii_uppercase();
            Ok(NormaliserChunk::Text(text))
        }
        "cardinal" => {
            let num = text.trim().parse::<i64>()?;
            let text = Num2Words::new(num)
                .cardinal()
                .to_words()
                .map_err(|e| anyhow::anyhow!(e))?
                .replace("-", " ")
                .to_ascii_uppercase();
            Ok(NormaliserChunk::Text(text))
        }
        "characters" => {
            let characters = text.graphemes(true).collect::<Vec<&str>>().join(" ");
            Ok(NormaliserChunk::Text(normalise_text(&characters)))
        }
        s => {
            anyhow::bail!("Unsupported say-as: {}", s);
        }
    }
}

pub fn normalise_ssml(x: &str) -> anyhow::Result<NormalisedText> {
    let parser = SsmlParserBuilder::default().expand_sub(true).build()?;

    let mut res = NormalisedText::default();
    let mut stack = vec![];
    let mut push_text = true;
    for event in parser.parse(x)?.event_iter() {
        match event {
            ParserEvent::Text(t) => {
                if push_text {
                    res.chunks.push(NormaliserChunk::Text(normalise_text(&t)));
                } else if let Some(tag) = stack.last() {
                    // We should look at the stack to see if there's something we're meant to be
                    // doing
                    match tag {
                        ParsedElement::SayAs(sa) => {
                            res.chunks.push(handle_say_as(sa, &t)?);
                        }
                        ParsedElement::Phoneme(ph) => {
                            if matches!(res.chunks.last(), Some(NormaliserChunk::Pronunciation(_)))
                            {
                                debug!(
                                    "Skipping: {} because we already pushed phonemes {:?}",
                                    t, ph
                                );
                            } else {
                                warn!("Couldn't handle phoneme tag, trying to just normalise!");
                            }
                        }
                        _ => unreachable!(),
                    }
                } else {
                    warn!("I don't know what to do with myself");
                }
            }
            ParserEvent::Open(open) => {
                match &open {
                    ParsedElement::SayAs(_) => {
                        push_text = false;
                    }
                    ParsedElement::Phoneme(ph) => {
                        push_text = false;
                        if matches!(ph.alphabet, None | Some(PhonemeAlphabet::Ipa)) {
                            let pronunciation = ipa_string_to_units(&ph.ph);
                            res.chunks
                                .push(NormaliserChunk::Pronunciation(pronunciation));
                        }
                    }
                    e => {
                        error!("Unhandled open tag: {:?}", e);
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
                ParsedElement::Break(ba) => {
                    let duration = match (ba.time.map(|x| x.duration()), ba.strength) {
                        (Some(duration), _) => duration,
                        (_, Some(strength)) => match strength {
                            Strength::No => continue,
                            Strength::ExtraWeak => Duration::from_secs_f32(0.2),
                            Strength::Weak => Duration::from_secs_f32(0.5),
                            Strength::Medium => Duration::from_secs(1),
                            Strength::Strong => Duration::from_secs(2),
                            Strength::ExtraStrong => Duration::from_secs(5),
                        },
                        _ => Duration::from_secs(1),
                    };
                    res.chunks.push(NormaliserChunk::Break(duration));
                }
                _ => {
                    error!("Unhandled tag: {:?}", tag);
                }
            },
        }
    }
    Ok(res)
}

pub fn normalise_text(x: &str) -> String {
    static IS_NUM: OnceCell<Regex> = OnceCell::new();
    static IS_PUNCT: OnceCell<Regex> = OnceCell::new();
    static NUM_SPLITTER: OnceCell<Regex> = OnceCell::new();

    let is_num = IS_NUM.get_or_init(|| Regex::new(r#"\d"#).unwrap());
    let is_punct = IS_PUNCT.get_or_init(|| Regex::new(r#"[[:punct:]]$"#).unwrap());
    let num_splitter = NUM_SPLITTER.get_or_init(|| {
        Regex::new("(?<head>[[:alpha:]]*)(?<digit>[[:digit:]]*)(?<tail>[[:alpha:]]*)").unwrap()
    });

    let mut result = String::new();
    let s = deunicode(x);

    let mut words: Vec<String> = s
        .split_ascii_whitespace()
        .map(|x| x.to_string())
        .collect::<Vec<_>>();

    while !words.is_empty() {
        let word = words.remove(0);

        // So NAN is a number... Be careful! https://github.com/Ballasi/num2words/issues/12
        let mut end_punct = None;
        let word = if let Some(punct) = is_punct.find(&word) {
            if let Ok(punct) = Punctuation::from_str(punct.as_str()) {
                end_punct = Some(punct);
            } else {
                info!("Unhandled punctuation: {}", punct.as_str());
            }
            &word[0..punct.start()]
        } else {
            &word
        };

        // Now we want to put some distance between words and letters to make 3D and k8s work
        let mut word = word.to_string();
        word.retain(valid_char);
        word.make_ascii_uppercase();

        let after = num_splitter.replace_all(&word, "$head $digit $tail");
        let mut split = after.trim().split_ascii_whitespace().collect::<Vec<_>>();
        if split.len() > 1 {
            if let Some(p) = end_punct {
                words.insert(0, p.to_string());
            }
            for s in split.drain(..).rev() {
                words.insert(0, s.to_string());
            }
        } else {
            if is_num.is_match(&word) {
                if let Some(number) = Num2Words::parse(&word).and_then(|x| x.to_words().ok()) {
                    // This should hopefully never fail if it could parse it in the first place
                    result.push_str(&number.replace("-", " ").to_ascii_uppercase());
                } else {
                    let mut word = word.to_string();
                    word.retain(valid_char);
                    word.make_ascii_uppercase();
                    result.push_str(&word);
                }
            } else {
                result.push_str(&word);
            }
            if let Some(end_punct) = end_punct {
                // Push the punctuation back on
            }
            result.push(' ');
        }
    }
    if !result.is_empty() {
        let _ = result.pop();
    }
    // 3D turns to THREE - we need to fix that!
    debug!("output: {} {}", x, result);
    result
}

fn valid_char(x: char) -> bool {
    !r#"!"£$%^&*()-_=+[{]};:'@#~,<.>/?|\`¬"#.contains(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_text_norm() {
        assert_eq!(normalise_text("3d"), "THREE D");
        assert_eq!(normalise_text("2nd"), "SECOND");
        assert_eq!(normalise_text("k8s"), "K EIGHT S");
    }

    #[test]
    fn duplicate_removal() {
        assert_eq!(dict_normalise("BATH(2)"), "BATH");
        assert_eq!(dict_normalise("HELLO!(45)"), "HELLO");
        assert_eq!(dict_normalise("(3)d"), "3D");
    }

    #[test]
    fn ssml_text_normalisation() {
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
        <sub alias="World Wide Web Consortium">W3C</sub>#";
        </speak>"#;
        let expected = "WORLD WIDE WEB CONSORTIUM";

        assert_eq!(normalise_ssml(text).unwrap().text(), expected);

        let text = r#"<speak>
        <say-as interpret-as="characters">10</say-as>.
        </speak>"#;
        let expected = "ONE ZERO";

        assert_eq!(normalise_ssml(text).unwrap().text(), expected);
    }
}
