use crate::phonemes::*;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_onnx::onnx::ModelProto;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum PhonemeId {
    Pad,
    Unk,
    Phoneme(PhoneticUnit),
    Space,
    FullStop,
    Comma,
    QuestionMark,
    ExclamationMark,
    Dash,
}

fn generate_id_list() -> Vec<PhonemeId> {
    let mut res = vec![PhonemeId::Pad, PhonemeId::Unk];

    let phones = [
        "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0",
        "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1",
        "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH",
        "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH",
        "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
    ];

    res.extend(phones.map(|x| PhonemeId::Phoneme(PhoneticUnit::from_str(x).unwrap())));
    res.extend_from_slice(&[
        PhonemeId::Space,
        PhonemeId::FullStop,
        PhonemeId::Comma,
        PhonemeId::QuestionMark,
        PhonemeId::ExclamationMark,
        PhonemeId::Dash,
    ]);

    res
}

pub struct SpeedySpeech {
    model_proto: ModelProto,
    phoneme_ids: Vec<PhonemeId>,
}

impl SpeedySpeech {
    #[must_use]
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        // read all for debugging
        let model_proto = candle_onnx::read_file(path.as_ref()).unwrap();
        let graph = match model_proto.graph.as_ref() {
            None => anyhow::bail!("No graph included in ONNX"),
            Some(graph) => graph,
        };
        for input in &graph.input {
            println!("Graph input: {:?}", input);
        }
        for output in &graph.output {
            println!("Graph output: {:?}", output);
        }
        Ok(Self {
            model_proto,
            phoneme_ids: generate_id_list(),
        })
    }

    pub fn infer(&self) -> anyhow::Result<()> {
        let inputs = HashMap::new();
        candle_onnx::simple_eval(&self.model_proto, inputs)?;
        todo!()
    }

    fn best_match_for_unit(&self, unit: PhoneticUnit) -> usize {
        let mut best = 2; // UNK
        for (i, potential) in self
            .phoneme_ids
            .iter()
            .enumerate()
            .filter(|(_, x)| matches!(x, PhonemeId::Phoneme(v) if v.phone == unit.phone))
        {
            if best == 2 {
                best = i;
            }
            if let PhonemeId::Phoneme(v) = potential {
                if unit.context.is_none() {
                    best = i;
                    break;
                } else if *v == unit {
                    best = i;
                    break;
                }
            }
        }
        best
    }
}
