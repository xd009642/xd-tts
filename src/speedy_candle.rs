use crate::phonemes::*;
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

fn generate_id_list() -> Vec<Unit> {
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

pub struct SpeedySpeech {
    model_proto: ModelProto,
    phoneme_ids: Vec<Unit>,
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

    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f64>> {
        let graph = self.model_proto.graph.as_ref().unwrap();

        let mut inputs = HashMap::new();

        for input in graph.input.iter() {
            let value = if input.name == "phonemes" {
                // Phonemes is a sequence tensor of [batch_size, phonemes]
                let phonemes = units
                    .iter()
                    .map(|x| self.best_match_for_unit(x))
                    .collect::<Vec<_>>();
                Tensor::from_vec(phonemes, (1, units.len()), &Device::Cpu)?
            } else if input.name == "plen" {
                Tensor::from_iter([units.len() as i64], &Device::Cpu)?
            } else {
                anyhow::bail!("Unexpected input: {:?}", input);
            };
            inputs.insert(input.name.clone(), value);
        }

        let result = candle_onnx::simple_eval(&self.model_proto, inputs)?;
        // So lets just get rid of the phoneme durations since I don't care for them
        if let Some(spectrogram) = result.get("spec") {
            // We want to remove batch dimension and then transpose the matrix/invert whatever
            todo!()
        } else {
            anyhow::bail!("No spectrogram provided on output!");
        }
    }

    fn best_match_for_unit(&self, unit: &Unit) -> i64 {
        if let Unit::Phone(unit) = unit {
            let mut best = 2; // UNK
            for (i, potential) in self
                .phoneme_ids
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
            self.phoneme_ids
                .iter()
                .enumerate()
                .find(|(_, x)| *x == unit)
                .map(|(i, _)| i as i64)
                .unwrap_or(2)
        }
    }
}
