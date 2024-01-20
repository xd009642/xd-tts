use anyhow::Error;
use std::fmt;
use std::str::FromStr;
use tracing::{error, warn};
use unicode_segmentation::UnicodeSegmentation;

pub type Pronunciation = Vec<PhoneticUnit>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Unit {
    /// An ARPA phoneme
    Phone(PhoneticUnit),
    /// Unknown
    Unk,
    /// Space
    Space,
    /// Punctuation
    Punct(Punctuation),
    /// A character - useful for models that have a character to ID mapping
    Character(char),
    /// Padding character
    Padding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Punctuation {
    FullStop,
    Comma,
    QuestionMark,
    ExclamationMark,
    Dash,
    OpenBracket,
    CloseBracket,
    Colon,
    SemiColon,
    Apostrophe,
}

impl Punctuation {
    pub fn is_sentence_end(&self) -> bool {
        matches!(
            self,
            Self::FullStop | Self::QuestionMark | Self::ExclamationMark
        )
    }

    pub fn is_pause(&self) -> bool {
        self.is_sentence_end() || matches!(self, Self::Comma | Self::SemiColon)
    }
}

pub fn ipa_to_unit(ipa: &str) -> anyhow::Result<Unit> {
    let phone = match ipa {
        "ɒ" | "ɑ" => ArpaPhone::Aa,
        "æ" => ArpaPhone::Ae,
        "ʌ" => ArpaPhone::Ah,
        "ɔ" => ArpaPhone::Ao,
        "aʊ" => ArpaPhone::Aw,
        //"ə" => ArpaPhone::Ax,
        "aɪ" => ArpaPhone::Ay,
        "ɛ" => ArpaPhone::Eh,
        "ɝ" => ArpaPhone::Er,
        "eɪ" => ArpaPhone::Ey,
        "ɪ" => ArpaPhone::Ih,
        //"ɨ" => ArpaPhone::Ix,
        "i" => ArpaPhone::Iy,
        "oʊ" => ArpaPhone::Ow,
        "ɔɪ" => ArpaPhone::Oy,
        "ʊ" => ArpaPhone::Uh,
        "u" => ArpaPhone::Uw,
        //"ʉ" => ArpaPhone::Ux,
        "b" => ArpaPhone::B,
        "tʃ" => ArpaPhone::Ch,
        "d" => ArpaPhone::D,
        "ð" => ArpaPhone::Dh,
        //"ɾ" => ArpaPhone::Dx,
        "f" => ArpaPhone::F,
        "ɡ" => ArpaPhone::G,
        "h" => ArpaPhone::Hh,
        "dʒ" => ArpaPhone::Jh,
        "k" => ArpaPhone::K,
        "l" => ArpaPhone::L,
        "m" => ArpaPhone::M,
        "n" => ArpaPhone::N,
        "ŋ" => ArpaPhone::Ng,
        //"ɾ̃" => ArpaPhone::Nx,
        "p" => ArpaPhone::P,
        //"ʔ" => ArpaPhone::Q,
        "ɹ" => ArpaPhone::R,
        "s" => ArpaPhone::S,
        "ʃ" => ArpaPhone::Sh,
        "t" => ArpaPhone::T,
        "θ" => ArpaPhone::Th,
        "v" => ArpaPhone::V,
        "w" => ArpaPhone::W,
        //"ʍ" => ArpaPhone::Wh,
        "j" => ArpaPhone::Y,
        "z" => ArpaPhone::Z,
        "ʒ" => ArpaPhone::Zh,
        _ => anyhow::bail!("unsupported/invalid IPA Phone {}", ipa),
    };
    Ok(Unit::Phone(PhoneticUnit {
        phone,
        context: None,
    }))
}

pub fn ipa_string_to_units(ipa: &str) -> Vec<Unit> {
    let get_unit = |g: &str| {
        if g.trim().is_empty() {
            Unit::Space
        } else {
            match ipa_to_unit(g) {
                Ok(s) => s,
                Err(e) => {
                    error!("Failed to map phoneme pushing unk: {}", e);
                    Unit::Unk
                }
            }
        }
    };

    let mut res = vec![];
    let mut graphemes = ipa.graphemes(true).collect::<Vec<&str>>();
    let mut buffer = String::new();
    for g in graphemes.drain(..) {
        if buffer.is_empty() {
            if matches!(g, "t" | "a" | "d") {
                buffer.push_str(g);
            } else {
                res.push(get_unit(g));
            }
        } else {
            let original = buffer.clone();
            buffer.push_str(g);
            match ipa_to_unit(&buffer) {
                Ok(s) => {
                    res.push(s);
                    buffer.clear();
                }
                Err(_) => {
                    buffer.clear();
                    res.push(get_unit(&original));
                    if matches!(g, "t" | "a" | "d") {
                        buffer.push_str(g);
                    } else {
                        res.push(get_unit(g));
                    }
                }
            }
        }
    }
    res
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Phone(p) => write!(f, "{}", p),
            Self::Unk => write!(f, "<UNK>"),
            Self::Space => write!(f, " "),
            Self::Punct(p) => write!(f, "{}", p),
            Self::Padding => write!(f, "<PAD>"),
            Self::Character(c) => write!(f, "{}", c),
        }
    }
}

impl fmt::Display for Punctuation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::FullStop => write!(f, "."),
            Self::Comma => write!(f, ","),
            Self::QuestionMark => write!(f, "?"),
            Self::ExclamationMark => write!(f, "!"),
            Self::Dash => write!(f, "-"),
            Self::OpenBracket => write!(f, "("),
            Self::CloseBracket => write!(f, ")"),
            Self::Colon => write!(f, ":"),
            Self::SemiColon => write!(f, ";"),
            Self::Apostrophe => write!(f, "'"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PhoneticUnit {
    pub phone: ArpaPhone,
    pub context: Option<AuxiliarySymbol>,
}

impl fmt::Display for PhoneticUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.phone)?;
        if let Some(symbol) = self.context {
            write!(f, "{}", symbol)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
/// Get the descriptions from (here)[https://en.wikipedia.org/wiki/ARPABET], we're using 2 letter ARPABET  
pub enum ArpaPhone {
    /// Open central unrounded vowel or open back rounded vowel. The "al" in "balm" or "o" in
    /// "bot".
    Aa,
    /// Near-open front unrounded vowel. The "a" in "bat".
    Ae,
    /// Near-open central vowel. The "u" in "butt".
    Ah,
    ///
    Ao,
    ///
    Aw,
    ///
    Ay,
    ///
    B,
    ///
    Ch,
    ///
    D,
    ///
    Dh,
    ///
    Eh,
    ///
    Er,
    Ey,
    F,
    G,
    Hh,
    Ih,
    Iy,
    Jh,
    K,
    L,
    M,
    N,
    Ng,
    Ow,
    Oy,
    P,
    R,
    S,
    Sh,
    T,
    Th,
    Uh,
    Uw,
    V,
    W,
    Y,
    Z,
    Zh,
}

impl fmt::Display for ArpaPhone {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Aa => write!(f, "AA"),
            Self::Ae => write!(f, "AE"),
            Self::Ah => write!(f, "AH"),
            Self::Ao => write!(f, "AO"),
            Self::Aw => write!(f, "AW"),
            Self::Ay => write!(f, "AY"),
            Self::B => write!(f, "B"),
            Self::Ch => write!(f, "CH"),
            Self::D => write!(f, "D"),
            Self::Dh => write!(f, "DH"),
            Self::Eh => write!(f, "EH"),
            Self::Er => write!(f, "ER"),
            Self::Ey => write!(f, "EY"),
            Self::F => write!(f, "F"),
            Self::G => write!(f, "G"),
            Self::Hh => write!(f, "HH"),
            Self::Ih => write!(f, "IH"),
            Self::Iy => write!(f, "IY"),
            Self::Jh => write!(f, "JH"),
            Self::K => write!(f, "K"),
            Self::L => write!(f, "L"),
            Self::M => write!(f, "M"),
            Self::N => write!(f, "N"),
            Self::Ng => write!(f, "NG"),
            Self::Ow => write!(f, "OW"),
            Self::Oy => write!(f, "OY"),
            Self::P => write!(f, "P"),
            Self::R => write!(f, "R"),
            Self::S => write!(f, "S"),
            Self::Sh => write!(f, "SH"),
            Self::T => write!(f, "T"),
            Self::Th => write!(f, "TH"),
            Self::Uh => write!(f, "UH"),
            Self::Uw => write!(f, "UW"),
            Self::V => write!(f, "V"),
            Self::W => write!(f, "W"),
            Self::Y => write!(f, "Y"),
            Self::Z => write!(f, "Z"),
            Self::Zh => write!(f, "ZH"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum AuxiliarySymbol {
    NoStress,
    PrimaryStress,
    SecondaryStress,
    TertiaryStress,
    Silence,
    NonSpeechSegment,
    MorphemeBoundary,
    WordBoundary,
    UtteranceBoundary,
    ToneGroupBoundary,
    FallingOrDecliningJuncture,
    RisingOrInternalJuncture,
    FallRiseOrNonTerminalJuncture,
}

impl fmt::Display for AuxiliarySymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::NoStress => write!(f, "0"),
            Self::PrimaryStress => write!(f, "1"),
            Self::SecondaryStress => write!(f, "2"),
            Self::TertiaryStress => write!(f, "3"),
            Self::Silence => write!(f, "-"),
            Self::NonSpeechSegment => write!(f, "!"),
            Self::MorphemeBoundary => write!(f, "+"),
            Self::WordBoundary => write!(f, "/"),
            Self::UtteranceBoundary => write!(f, "#"),
            Self::ToneGroupBoundary => write!(f, ":"),
            Self::FallingOrDecliningJuncture => write!(f, ":1"),
            Self::RisingOrInternalJuncture => write!(f, ":2"),
            Self::FallRiseOrNonTerminalJuncture => write!(f, ":3"),
        }
    }
}

impl FromStr for Unit {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let res = match s.trim() {
            "" if !s.is_empty() => Unit::Space,
            "." => Unit::Punct(Punctuation::FullStop),
            "," => Unit::Punct(Punctuation::Comma),
            "?" => Unit::Punct(Punctuation::QuestionMark),
            "!" => Unit::Punct(Punctuation::ExclamationMark),
            "-" => Unit::Punct(Punctuation::Dash),
            "(" => Unit::Punct(Punctuation::OpenBracket),
            ")" => Unit::Punct(Punctuation::CloseBracket),
            ";" => Unit::Punct(Punctuation::SemiColon),
            ":" => Unit::Punct(Punctuation::Colon),
            "'" => Unit::Punct(Punctuation::Apostrophe),
            "<PAD>" => Unit::Padding,
            "<UNK>" => Unit::Unk,
            trimmed => {
                // There is overlap with characters and ARPA phonemes. But here we're going to
                // prioritise ARPA!
                let unit_res = PhoneticUnit::from_str(trimmed);
                match unit_res {
                    Ok(unit) => Unit::Phone(unit),
                    Err(e) => {
                        let chars = trimmed.chars().collect::<Vec<_>>();
                        if chars.len() == 1 {
                            Unit::Character(chars[0])
                        } else {
                            return Err(e.context("failed to fallback to character unit"));
                        }
                    }
                }
            }
        };
        Ok(res)
    }
}

impl FromStr for Punctuation {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let res = match s.trim() {
            "." => Punctuation::FullStop,
            "," => Punctuation::Comma,
            "?" => Punctuation::QuestionMark,
            "!" => Punctuation::ExclamationMark,
            "-" => Punctuation::Dash,
            _ => {
                anyhow::bail!("Invalid punctuation: {}", s);
            }
        };
        Ok(res)
    }
}

impl FromStr for PhoneticUnit {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            Err(Error::msg("no symbols provided"))
        } else if s.len() > 4 {
            Err(Error::msg("input data too long"))
        } else if s.len() == 1 {
            let phone = ArpaPhone::from_str(s)?;
            Ok(Self {
                phone,
                context: None,
            })
        } else if s.len() == 4 {
            let phone = ArpaPhone::from_str(&s[..2])?;
            let context = AuxiliarySymbol::from_str(&s[2..])?;
            Ok(Self {
                phone,
                context: Some(context),
            })
        } else {
            let phone_syms = "AEHOWYBCDFGIJKLMNPRSUVZ";
            let stop = if phone_syms.contains(s.chars().nth(1).unwrap()) {
                2
            } else {
                1
            };
            let phone = ArpaPhone::from_str(&s[..stop])?;
            let context = if stop == s.len() {
                None
            } else {
                Some(AuxiliarySymbol::from_str(&s[stop..])?)
            };
            Ok(Self { phone, context })
        }
    }
}

impl FromStr for ArpaPhone {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "AA" => Ok(Self::Aa),
            "AE" => Ok(Self::Ae),
            "AH" => Ok(Self::Ah),
            "AO" => Ok(Self::Ao),
            "AW" => Ok(Self::Aw),
            "AY" => Ok(Self::Ay),
            "B" => Ok(Self::B),
            "CH" => Ok(Self::Ch),
            "D" => Ok(Self::D),
            "DH" => Ok(Self::Dh),
            "EH" => Ok(Self::Eh),
            "ER" => Ok(Self::Er),
            "EY" => Ok(Self::Ey),
            "F" => Ok(Self::F),
            "G" => Ok(Self::G),
            "HH" => Ok(Self::Hh),
            "IH" => Ok(Self::Ih),
            "IY" => Ok(Self::Iy),
            "JH" => Ok(Self::Jh),
            "K" => Ok(Self::K),
            "L" => Ok(Self::L),
            "M" => Ok(Self::M),
            "N" => Ok(Self::N),
            "NG" => Ok(Self::Ng),
            "OW" => Ok(Self::Ow),
            "OY" => Ok(Self::Oy),
            "P" => Ok(Self::P),
            "R" => Ok(Self::R),
            "S" => Ok(Self::S),
            "SH" => Ok(Self::Sh),
            "T" => Ok(Self::T),
            "TH" => Ok(Self::Th),
            "UH" => Ok(Self::Uh),
            "UW" => Ok(Self::Uw),
            "V" => Ok(Self::V),
            "W" => Ok(Self::W),
            "Y" => Ok(Self::Y),
            "Z" => Ok(Self::Z),
            "ZH" => Ok(Self::Zh),
            _ => {
                Err(Error::msg("invalid phone")
                    .context(format!("{} is not a valid ARPABET phone", s)))
            }
        }
    }
}

impl FromStr for AuxiliarySymbol {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(Self::NoStress),
            "1" => Ok(Self::PrimaryStress),
            "2" => Ok(Self::SecondaryStress),
            "3" => Ok(Self::TertiaryStress),
            "-" => Ok(Self::Silence),
            "!" => Ok(Self::NonSpeechSegment),
            "+" => Ok(Self::MorphemeBoundary),
            "/" => Ok(Self::WordBoundary),
            "#" => Ok(Self::UtteranceBoundary),
            ":" => Ok(Self::ToneGroupBoundary),
            ":1" | "." => Ok(Self::FallingOrDecliningJuncture),
            ":2" | "?" => Ok(Self::RisingOrInternalJuncture),
            ":3" => Ok(Self::FallRiseOrNonTerminalJuncture),
            _ => Err(Error::msg("invalid stress or auxiliary symbol")
                .context(format!("{} is not a valid symbol", s))),
        }
    }
}

pub fn best_match_for_unit(unit: &Unit, unit_list: &[Unit]) -> i64 {
    if let Unit::Phone(unit) = unit {
        let mut best = 2; // UNK
        for (i, potential) in unit_list
            .iter()
            .enumerate()
            .filter(|(_, x)| matches!(x, Unit::Phone(v) if v.phone == unit.phone))
        {
            if best == 2 {
                best = i as i64;
            }
            if let Unit::Phone(v) = potential {
                if unit.context.is_none() && v.context.is_some() {
                    warn!("Unstressed phone when stressed expected: {:?}", v.phone);
                    best = i as i64;
                    break;
                } else if v == unit {
                    best = i as i64;
                    break;
                }
            }
        }
        best
    } else {
        unit_list
            .iter()
            .enumerate()
            .find(|(_, x)| *x == unit)
            .map(|(i, _)| i as i64)
            .unwrap_or(2)
    }
}

fn split_score(unit: &Unit) -> usize {
    match unit {
        Unit::Punct(p) if p.is_sentence_end() => 3,
        Unit::Padding => 3,
        Unit::Punct(p) if p.is_pause() => 2,
        Unit::Space => 1,
        _ => 0, // Should not pause
    }
}

/// Given a length constraint uses some heuristics to attempt to split up the transcript into
/// smaller chunks we can process. This function probably does a bit too much sorting and too many
/// vectors. But given how long the mel generation takes it's a drop in the pond!
pub fn find_splits(units: &[Unit], max_size: usize) -> Vec<usize> {
    let punct_and_spaces = units
        .iter()
        .enumerate()
        .map(|(i, x)| (i, split_score(x)))
        .filter(|(_, score)| *score > 0)
        .collect::<Vec<_>>();

    let mut results: Vec<usize> = punct_and_spaces
        .iter()
        .filter(|(_, score)| *score > 2)
        .map(|(i, _)| *i)
        .collect();
    // If there's just a single full stop at the end we need a delta to subtract against. This
    // ensures we always check at least len - 0 in the loop.
    results.insert(0, 0);

    let mut threshold_score = 1;
    let mut scan = true;
    let mut new_indexes = vec![];
    while scan {
        scan = false;
        let mut last_ref = units.len();
        for index in results.iter().rev() {
            if last_ref - index > max_size {
                scan = true;
                // split further
                new_indexes.extend(
                    punct_and_spaces
                        .iter()
                        .filter(|(i, score)| *i < last_ref && i > index && *score > threshold_score)
                        .map(|(i, _)| *i),
                );
            }
            last_ref = *index;
        }
        if scan {
            results.append(&mut new_indexes);
            results.sort_unstable();
        }
        if threshold_score > 0 {
            threshold_score -= 1;
        } else {
            scan = false;
        }
    }
    // Now we should merge things we broke up too small

    let mut merged_results = vec![];

    let mut running_total = 0;
    let mut last_insert = 0;

    for i in &results {
        if (i - last_insert) + running_total > max_size {
            merged_results.push(last_insert);
            running_total = i - last_insert;
        } else {
            running_total += i - last_insert;
        }
        last_insert = *i;
    }
    if running_total + (units.len() - last_insert) > max_size {
        if let Some(x) = results.last() {
            merged_results.push(*x);
        }
    }
    merged_results.dedup();

    merged_results
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ipa_remapping() {
        let ipa_str = "æɝtʃtdtdʒk";

        let ipa_converted = ipa_string_to_units(ipa_str);

        let arpa_parsed = "AE ER CH T D T JH K"
            .split_ascii_whitespace()
            .map(|x| Unit::from_str(x).unwrap())
            .collect::<Vec<Unit>>();

        assert_eq!(ipa_converted, arpa_parsed);
    }
}
