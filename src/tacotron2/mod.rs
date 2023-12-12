use crate::phonemes::*;
use std::str::FromStr;
use tch::{CModule, Tensor};
use ndarray::Array2;
use std::path::Path;
use tracing::info;

fn generate_id_list() -> Vec<Unit> {
    let phones = [
        "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2", "AO",
        "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2", "B", "CH", "D",
        "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1", "EY2", "F",
        "G", "HH", "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N",
        "NG", "OW", "OW0", "OW1", "OW2", "OY", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH",
        "UH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
    ];

    let mut res = vec![
        Unit::Padding,
        Unit::Punct(Punctuation::Dash),
        Unit::Punct(Punctuation::ExclamationMark),
        Unit::Punct(Punctuation::Apostrophe),
        Unit::Punct(Punctuation::OpenBracket),
        Unit::Punct(Punctuation::CloseBracket),
        Unit::Punct(Punctuation::Comma),
        Unit::Punct(Punctuation::FullStop),
        Unit::Punct(Punctuation::Colon),
        Unit::Punct(Punctuation::SemiColon),
        Unit::Punct(Punctuation::QuestionMark),
        Unit::Space,
    ];
    let characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        .chars()
        .map(|x| Unit::Character(x));

    res.extend(characters);
    res.extend(phones.iter().map(|x| Unit::from_str(x).unwrap()));

    res
}

// https://catalog.ngc.nvidia.com/orgs/nvidia/models/tacotron2pyt_jit_fp16/files

pub struct Tacotron2 {
    model: CModule,
    phoneme_ids: Vec<Unit>,
}


impl Tacotron2 {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model = CModule::load(path)?;
        Ok(Self {
            model,
            phoneme_ids: generate_id_list(),
        })
    }

    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        let phonemes = units
            .iter()
            .map(|x| best_match_for_unit(x, &self.phoneme_ids))
            .collect::<Vec<_>>();

        let phonemes = Tensor::f_of_slice(phonemes.as_slice())?.unsqueeze(0);

        let prediction = self.model.forward_ts(&[phonemes])?;

        info!("Prediction shape: {:?}", prediction.size());

        todo!()
    }
}
