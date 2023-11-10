use crate::phonemes::*;
use std::str::FromStr;

pub mod wonnx;
pub mod candle;

pub use candle::*;
pub use wonnx::*;

pub(crate) fn generate_id_list() -> Vec<Unit> {
    let mut res = vec![Unit::Padding, Unit::Unk];

    let phones = [
        "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0",
        "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1",
        "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH",
        "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH",
        "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
    ];

    res.extend(phones.map(|x| Unit::from_str(x).unwrap()));
    res.extend_from_slice(&[
        Unit::Space,
        Unit::FullStop,
        Unit::Comma,
        Unit::QuestionMark,
        Unit::ExclamationMark,
        Unit::Dash,
    ]);

    res
}

pub(crate) fn best_match_for_unit(unit: &Unit, unit_list: &[Unit]) -> i64 {
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
                    println!("Unstressed phone when stressed expected: {:?}", v.phone);
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
