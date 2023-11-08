use anyhow::Error;
use std::str::FromStr;

pub type Pronunciation = Vec<PhoneticUnit>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PhoneticUnit {
    pub phone: ArpaPhone,
    pub context: Option<AuxiliarySymbol>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
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
