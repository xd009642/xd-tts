//! The text normaliser is the first stage of a TTS pipeline, here we take the users text input and
//! convert it to an unambiguous form so audio can be generated - also known as an orthographic
//! form.
//!
//! For an example consider:
//!
//! > In 1970 £1970 had much higher spending power.
//!
//! As an orthographic transcript this would be written as:
//!
//! > In nineteen seventy one thousand nine hundred and seventy pounds had much higher spending
//! power
//!
//! This gets more complicated as we consider all the ways in which language is fluid and
//! ambiguous. Our eventual aim is to map these words into a list of phonemes and pauses so we can
//! generate audio from them. This means we eventually need to consider the meaning of the
//! sentence. For a perfect solution
//!
//! > You can lead a horse to water.
//!
//! > That went down like a lead balloon
//!
//! Here lead is a homograph, the same spelling but not necessarily the same meaning or
//! pronunciation. Given the wider context of the sentence we can as English speakers see how to
//! pronounce it. The challenge is making our TTS systems do this. Existing approaches generally
//! fall into two categories:
//!
//! 1. Rule based
//! 2. Data driven
//!
//! A rule based system would involve linguists and programmers working together to craft rules to
//! pick the right words, expansions and meanings to normalise to. A data driven approach would
//! utilise statistical or deep learning models to achieve the same effect. A system may also be a
//! hybrid system utilising statistical models in some places and rules in others to try and create
//! the best system given time/data/complexity of the given linguistics.
//!
//! For this project we won't be dealing with homographs and selecting the right pronunciation.
//! We'll instead make a simple rules based engine and try to handle numbers and unicode somewhat
//! correctly.
//!
//! ## SSML
//!
//! Speech Synthesis Markup Language is a way to guide a TTS system. It can affect such things as:
//!
//! 1. Language
//! 2. Prosody
//! 3. Text normalisation
//! 4. Pronunciation
//!
//! Given the complexities of text normalisation, commercial TTS systems often provide an SSML
//! interface so users can express their transcripts unambiguously and limit issues caused by
//! incorrect normalisation.
//!
//! We provide an SSML processor, and also handle normalisation of synthesisable text within the
//! SSML document. Because of this the `NormalisedText` struct is needed for the output of the
//! normaliser to show when elements from SSML i.e. breaks/pronunciation/prosody are active.
//!
//! The SSML spec is much wider than the features supported here, but generally it says a
//! synthesiser can quietly ignore features it doesn't support.
//!
//! ## A Note on Other Languages
//!
//! These docs and this example are largely only applicable to English for other languages there
//! are extra challenges in tokenisation and normalisation. There are also some other issues that
//! exist with English TTS systems that we haven't discussed. I'll attempt to highlight some of
//! these below.
//!
//! Code-switching, sometimes words from different languages may appear in text. We can see this in
//! English where some French phrases have been loaned or entered common speech i.e. "c'est la
//! vie". Likewise we may find proper nouns such as company names appearing in text with different
//! writing formats creating a mix between the two. For these rendering the text into speech as a
//! native speaker would say it can be very challenging.
//!
//! Arabic as a language has extra complexities because of diacritics, these impact the
//! pronunciation of the words a lot and are typically omitted in the written form. Arabic TTS
//! systems need to reinsert the diacritic marks which also vary by dialect further complicating
//! work if you want to support a specific accent/dialect instead of MSA. Also as spoken Arabic
//! isn't standardised users may not input in Modern Standard Arabic (MSA) and you may have to try
//! and diacritise words which have a different spelling in the users dialect compared to MSA.
//!
//! Japanese mixes kanji, hiragana and katakana in it's text - kanji are logographic (each kanji is
//! a whole word), and hiragana/katakana are syllabic. A sentence in Japanese will often use all 3
//! together. Getting the correct pronunciation of a kanji often relies on context, and even
//! hiragana. The hiragana はis pronounced as "ha", but if it's used to indicate a subject it
//! should be pronounced as "wa". Part of inferring the context requires splitting the sentence
//! into words, but written japanese doesn't use spaces to separate words further complicating the
//! implementation of text normalisation!
//!
//! There are undoubtedly many more examples spanning all languages, but these are examples from
//! two languages I've had experience with in my personal and professional life!
use crate::phonemes::Unit as TtsUnit;
use crate::phonemes::*;
use crate::CmuDictionary;
use deunicode::deunicode;
use num2words::Num2Words;
use once_cell::sync::OnceCell;
use regex::Regex;
use ssml_parser::{elements::*, parser::SsmlParserBuilder, ParserEvent};
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, error, warn};
use unicode_segmentation::UnicodeSegmentation;

/// A chunk of data that can be processed altogether by the TTS system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NormaliserChunk {
    /// Text to be spoken, this is normalised so should only contain a-z and punctuation
    Text(String),
    /// A pause in the speech.
    Break(Duration),
    /// An exact set of phonemes to be spoken.
    Pronunciation(Vec<TtsUnit>),
    /// Punctuation to be applied. This is separate so we can map it to pauses (if not handled by
    /// the model).
    Punct(Punctuation),
}

/// Output from the text normaliser, this contains a sequence of chunks to be processed. We return
/// this instead of the vector because:
///
/// 1. Ergonomics of methods on it
/// 2. Some languages may need more metadata - especially ones that undergo transliteration
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NormalisedText {
    /// Sequence of normalised chunks in the order they appear in the transcript.
    chunks: Vec<NormaliserChunk>,
}

impl NormalisedText {
    /// Takes the normaliser and a dictionary and converts all the text to an exact pronunciation.
    /// This does not handle picking the right pronunciation when there are multiple candidate
    /// ones, it will just select the first in the dictionary. Unsupported words will be skipped
    /// (traditionally there would be a G2P model to estimate a pronunciation for them).
    pub fn words_to_pronunciation(&mut self, dict: &CmuDictionary) {
        for x in self
            .chunks
            .iter_mut()
            .filter(|x| matches!(x, NormaliserChunk::Text(_)))
        {
            let units = match x {
                NormaliserChunk::Text(s) => {
                    let mut units = vec![];
                    for word in s.split_ascii_whitespace() {
                        if let Some(pronunciation) = dict.get_pronunciations(word) {
                            assert!(!pronunciation.is_empty());
                            debug!("{} is pronounced: {:?}", word, pronunciation);
                            units.extend(pronunciation[0].iter().map(|x| TtsUnit::Phone(*x)));
                            units.push(TtsUnit::Space);
                        } else {
                            warn!("Unsupported word: '{}'", word);
                        }
                    }
                    units
                }
                _ => unreachable!(),
            };
            *x = NormaliserChunk::Pronunciation(units);
        }
    }

    /// Converts the existing representation to be all in terms of `crate::phonemes::Unit`. This
    /// will turn words into a sequence of `Unit::Character` not convert to a pronunciation. If you
    /// want phonemes out use `NormalisedText::words_to_pronunciation`.
    pub fn convert_to_units(&mut self) {
        for x in self
            .chunks
            .iter_mut()
            .filter(|x| matches!(x, NormaliserChunk::Text(_) | NormaliserChunk::Punct(_)))
        {
            let units = match x {
                NormaliserChunk::Text(x) => {
                    let x = x.to_ascii_lowercase();
                    let mut chunk = vec![];
                    for c in x.chars() {
                        if c.is_whitespace() {
                            chunk.push(TtsUnit::Space);
                        } else if let Ok(punct) = Punctuation::from_str(c.to_string().as_str()) {
                            chunk.push(TtsUnit::Punct(punct));
                        } else {
                            chunk.push(TtsUnit::Character(c));
                        }
                    }
                    chunk
                }
                NormaliserChunk::Punct(p) => vec![TtsUnit::Punct(*p)],
                _ => unreachable!(),
            };
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
                NormaliserChunk::Text(s) => {
                    if !res.is_empty() && res.len() == res.trim_end().len() {
                        res.push(' ');
                    }
                    res.push_str(s)
                }
                NormaliserChunk::Punct(p) => res.push_str(&p.to_string()),
                NormaliserChunk::Pronunciation(_) => {
                    anyhow::bail!("Can't turn pronunciation chunk into text")
                }
                NormaliserChunk::Break(_) => {}
            }
        }
        Ok(res)
    }

    /// Works the same as to_string but panics if an element that can't be rendered to a string is
    /// present.
    pub fn to_string_unchecked(&self) -> String {
        self.to_string().unwrap()
    }

    /// Appends another set of normalised text chunks to this one.
    fn append(&mut self, mut other: NormalisedText) {
        self.chunks.append(&mut other.chunks);
    }
}

/// Runs text normalisation. Attempts to detect if the given transcript is SSML or just text and
/// pick an appropriate normaliser.
pub fn normalise(x: &str) -> anyhow::Result<NormalisedText> {
    // We're doing this really simply and assuming the start of a speak tag won't appear in
    // non-SSML.
    if x.contains("<speak") {
        normalise_ssml(x)
    } else {
        Ok(normalise_text(x))
    }
}

/// This is a normalisation just for CMU dictionary entries. These are typically words some
/// containing numbers - hence needing a mild normalisation. But also for words with multiple
/// entries they will add `(N)` after the word where N is the index of the pronunciation. This
/// normaliser removes that and then runs the standard normaliser on it.
pub fn dict_normalise(x: &str) -> String {
    // This regex is just for duplicate pronunciations in CMU dict
    static VERSION_REGEX: OnceCell<Regex> = OnceCell::new();
    let version_regex = VERSION_REGEX.get_or_init(|| Regex::new(r#"\(\d+\)$"#).unwrap());

    let version_strip = version_regex.replace_all(x, "");

    normalise_text(&version_strip).to_string_unchecked()
}

/// Handles an SSML `<say-as>` tag. This tag is used to help disambiguate numbers, make acronyms a
/// bit better to handle among other things. I've kept say-as support minimal but you could add as
/// many or little as you desire. There's also minimal validation that the input is correct instead
/// trying to do a best effort guess of what the user wants.
fn handle_say_as(say_as: &SayAsAttributes, text: &str) -> anyhow::Result<NormaliserChunk> {
    match say_as.interpret_as.as_str() {
        "ordinal" => {
            let num = text.trim().parse::<i64>()?;
            let text = Num2Words::new(num)
                .ordinal()
                .to_words()
                .map_err(|e| anyhow::anyhow!(e))?
                .replace('-', " ")
                .to_ascii_uppercase();
            Ok(NormaliserChunk::Text(text))
        }
        "cardinal" => {
            let num = text.trim().parse::<i64>()?;
            let text = Num2Words::new(num)
                .cardinal()
                .to_words()
                .map_err(|e| anyhow::anyhow!(e))?
                .replace('-', " ")
                .to_ascii_uppercase();
            Ok(NormaliserChunk::Text(text))
        }
        "characters" => {
            let characters = text.graphemes(true).collect::<Vec<&str>>().join(" ");
            let mut chunk = normalise_text(&characters);
            chunk
                .chunks
                .retain(|x| matches!(x, NormaliserChunk::Text(t) if !t.is_empty()));
            if chunk.chunks.len() == 1 {
                Ok(chunk.chunks.remove(0))
            } else {
                Ok(NormaliserChunk::Text(chunk.to_string_unchecked()))
            }
        }
        s => {
            anyhow::bail!("Unsupported say-as: {}", s);
        }
    }
}

/// Normalise SSML, here we iterate over the XML events and process them accordingly, this may be
/// expanded on for more elements as things start to take shape but initially it's being kept as
/// simple as possible. The SSML parser crate should remove non-synthesisable text by default and
/// this helps simplify some of our usage code!
pub fn normalise_ssml(x: &str) -> anyhow::Result<NormalisedText> {
    let parser = SsmlParserBuilder::default().expand_sub(true).build()?;

    let mut res = NormalisedText::default();
    let mut stack = vec![];
    // Some of the tags we support mean we ignore the text inside the tag and instead use XML
    // attributes to work out pronunciation. Hence the need to track push_text
    let mut push_text = true;
    for event in parser.parse(x)?.event_iter() {
        match event {
            ParserEvent::Text(t) => {
                if push_text {
                    res.append(normalise_text(&t));
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
                    ParsedElement::Speak(_) => {}
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
                ParsedElement::Break(b) => {
                    let duration = match (b.time.map(|x| x.duration()), b.strength) {
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

/// Numbers are quite complicated. Here we have basic handling for ordinals, cardinals and numbers
/// with letters or symbols after them. Currency, years, phone numbers all add extra complexity and
/// have been ignored. So if you input a phone number like 0800001066 it will read it as a number -
/// not an intuitive way to receive a phone number!
fn process_number(x: &str) -> anyhow::Result<String> {
    static IS_ORDINAL: OnceCell<Regex> = OnceCell::new();
    static JUST_NUMBER: OnceCell<Regex> = OnceCell::new();
    static NUM_SPLITTER: OnceCell<Regex> = OnceCell::new();

    let is_ordinal = IS_ORDINAL.get_or_init(|| Regex::new("^[[:digit:]](st|nd|th|rd)$").unwrap());
    let just_number = JUST_NUMBER.get_or_init(|| Regex::new(r#"^[\d\.,]+$"#).unwrap());
    let num_splitter = NUM_SPLITTER
        .get_or_init(|| Regex::new(r#"(?<head>\D*)(?<digit>[[:digit:]]+)(?<tail>\D*)"#).unwrap());

    if is_ordinal.is_match(x) {
        let text = Num2Words::parse(x)
            .and_then(|x| x.ordinal().to_words().ok())
            .ok_or_else(|| anyhow::anyhow!("Invalid ordinal: '{}'", x))?
            .replace('-', " ")
            .to_ascii_uppercase();
        Ok(text)
    } else if just_number.is_match(x) {
        let text = Num2Words::parse(x)
            .and_then(|x| x.to_words().ok())
            .ok_or_else(|| anyhow::anyhow!("Invalid number '{}'", x))?
            .replace('-', " ")
            .to_ascii_uppercase();

        Ok(text)
    } else if let Some(cap) = num_splitter.captures(x) {
        // We can to_string the normalise text stuff here because we know that this is isolated to
        // a single word and punctuation has already been stripped.
        let head = normalise_text(&cap["head"]).to_string_unchecked();

        let digit = Num2Words::parse(&cap["digit"])
            .and_then(|x| x.to_words().ok())
            .ok_or_else(|| anyhow::anyhow!("Invalid number: '{}'", &cap["digit"]))?
            .replace('-', " ")
            .to_ascii_uppercase();

        let tail = normalise_text(&cap["tail"]).to_string_unchecked();

        let mut res = String::new();

        let head_t = head.trim();
        let tail_t = tail.trim();

        if !head_t.is_empty() {
            res.push_str(head_t);
            res.push(' ');
        }

        res.push_str(&digit);

        if !tail_t.is_empty() {
            res.push(' ');
            res.push_str(tail_t);
        }

        Ok(res)
    } else {
        unreachable!()
    }
}

/// Normalise non-SSML text, this splits by words and then attempts to normalise each word in
/// isolation as well as gathering the punctuation information.
pub fn normalise_text(x: &str) -> NormalisedText {
    static IS_NUM: OnceCell<Regex> = OnceCell::new();
    static IS_PUNCT: OnceCell<Regex> = OnceCell::new();
    static PROBLEM_CHARS: OnceCell<Regex> = OnceCell::new();

    let is_num = IS_NUM.get_or_init(|| Regex::new(r#"\d"#).unwrap());
    let is_punct = IS_PUNCT.get_or_init(|| Regex::new(r#"[[:punct:]]$"#).unwrap());
    let problem_chars = PROBLEM_CHARS.get_or_init(|| Regex::new(r#"[\[\(\)\]\-:]"#).unwrap());

    let mut text_buffer = String::new();
    let mut result = NormalisedText::default();
    let s = deunicode(x);

    // Lets initially clean away some problem characters! This is a bit of a hack. And also ones
    // like `-` may be spoken or not.
    let s = problem_chars.replace_all(&s, " ");

    let mut words: Vec<String> = s
        .split_ascii_whitespace()
        .map(|x| x.to_string())
        .collect::<Vec<_>>();

    while !words.is_empty() {
        let mut word = words.remove(0);

        if word.trim() == "&" {
            word = word.replace('&', "and");
        }

        // So NAN is a number... Be careful! https://github.com/Ballasi/num2words/issues/12
        let mut end_punct = None;
        let word = if let Some(punct) = is_punct.find(&word) {
            if let Ok(punct) = Punctuation::from_str(punct.as_str()) {
                end_punct = Some(punct);
            } else if !matches!(punct.as_str(), "'" | "\"") {
                // We can ignore apostrophes!
                warn!("Unhandled punctuation: '{}'", punct.as_str());
            }
            &word[0..punct.start()]
        } else {
            &word
        };

        if is_num.is_match(word) {
            // We don't want to remove spaces after punctuation!
            text_buffer.push_str(&process_number(word).unwrap());
        } else {
            let mut word = word.to_string();
            word.retain(valid_char);
            word.make_ascii_uppercase();
            // We don't want to remove spaces after punctuation!
            text_buffer.push_str(&word);
        }
        if let Some(end_punct) = end_punct {
            // Push the punctuation back on
            if !text_buffer.is_empty() {
                result
                    .chunks
                    .push(NormaliserChunk::Text(text_buffer.clone()));
                text_buffer.clear();
            }
            result.chunks.push(NormaliserChunk::Punct(end_punct));
            // Keeps space after punct
            text_buffer.push(' ');
        } else {
            text_buffer.push(' ');
        }
    }
    if !text_buffer.is_empty() {
        let _ = text_buffer.pop();
        if !text_buffer.is_empty() {
            result.chunks.push(NormaliserChunk::Text(text_buffer));
        }
    }
    debug!("output: {} {:?}", x, result);
    result
}

/// Used to remove characters we can't synthesise from words. Any punctuation in here should be
/// picked up and added to the normaliser output before we strip it!
fn valid_char(x: char) -> bool {
    !r#"!"£$%^&*()-_=+[{]};:'@#~,<.>/?|\`¬"#.contains(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_text_norm() {
        assert_eq!(normalise_text("2nd").to_string_unchecked(), "SECOND");
        assert_eq!(normalise_text("3d").to_string_unchecked(), "THREE D");
        assert_eq!(normalise_text("k8s").to_string_unchecked(), "K EIGHT S");
    }

    #[test]
    fn duplicate_removal() {
        assert_eq!(dict_normalise("BATH(2)"), "BATH");
        assert_eq!(dict_normalise("HELLO(45)"), "HELLO");
        assert_eq!(dict_normalise("(3)d"), "THREE D");
    }

    #[test]
    fn hyphened_numbers() {
        assert_eq!(
            normalise_text("sixty-four ninety-three").to_string_unchecked(),
            "SIXTY FOUR NINETY THREE"
        )
    }

    #[test]
    fn extract_punctuation() {
        let actual = normalise_text("Is this my 1st talk? You tell me!");

        let expected = NormalisedText {
            chunks: vec![
                NormaliserChunk::Text("IS THIS MY FIRST TALK".to_string()),
                NormaliserChunk::Punct(Punctuation::QuestionMark),
                NormaliserChunk::Text(" YOU TELL ME".to_string()),
                NormaliserChunk::Punct(Punctuation::ExclamationMark),
            ],
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn ssml_text_normalisation() {
        let text = r#"<speak>
        <say-as interpret-as="characters">SSML</say-as> 
        </speak>"#;
        let expected = "S S M L";

        assert_eq!(
            normalise_ssml(text).unwrap().to_string_unchecked(),
            expected
        );

        let text = r#"<speak>
        <say-as interpret-as="cardinal">10</say-as> 
        </speak>"#;
        let expected = "TEN";

        assert_eq!(
            normalise_ssml(text).unwrap().to_string_unchecked(),
            expected
        );

        let text = r#"<speak>
        <say-as interpret-as="ordinal">10</say-as>
        </speak>"#;
        let expected = "TENTH";

        assert_eq!(
            normalise_ssml(text).unwrap().to_string_unchecked(),
            expected
        );

        let text = r#"<speak>
        <sub alias="World Wide Web Consortium">W3C</sub>
        </speak>"#;
        let expected = "WORLD WIDE WEB CONSORTIUM";

        assert_eq!(
            normalise_ssml(text).unwrap().to_string_unchecked(),
            expected
        );

        let text = r#"<speak>
        <say-as interpret-as="characters">10</say-as>.
        </speak>"#;
        let expected = "ONE ZERO.";

        assert_eq!(
            normalise_ssml(text).unwrap().to_string_unchecked(),
            expected
        );
    }
}
