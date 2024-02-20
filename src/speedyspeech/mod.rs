//! The steps for speedyspeech were:
//!
//! 1. Download the latest model from [here](https://github.com/janvainer/speedyspeech/releases/download/v0.2/speedyspeech.pth)
//! 2. Convert to ONNX via my script in scripts/speedyspeech/onnx_experter.py
//!
//! Speedyspeech is similar in some senses to tacotron2 it's main components are:
//!
//! 1. A phoneme duration predictor
//! 2. An encoder network
//! 3. A decoder network
//!
//! The duration predictor is used to define the loop counts for the decoder, and this
//! internal variable dimension is something that is unlikely to work in most ONNX runtimes.
//! Instead what should be done to make it more exportable is splitting these three sub-networks
//! into three individual networks and figuring out the changes that need to be made to the decoder
//! network to do inference. This is the process that Nvidia did in the tacotron2 repo and
//! something I decided not to do myself as it would be a lot of time sunken into something that
//! might work poorly.
//!
//! If someones interested in trying that out, they can either adapt the speedyspeech repo, or
//! attempt to use something like onnx graph surgeon to mutate the graph I output. I have low hope
//! in adapting the repo however because...
//!
//! # Problems
//!
//! After getting our ONNX file however it uses the loop operator and
//! has dynamically sized inputs within the model. These are two things that
//! proved fatal to running it in a Rust runtime.
//!
//! After finding out about ORT I tried it there, unfortunately pytorch can output ONNX that
//! doesn't obey the standard (-1 dimension size in expand nodes). 
//!
//! There's an old version of torch used in speedyspeech, one with worse ONNX support. Any changes
//! to the pretrained graph would need to be done via this old version of torch and torch doesn't
//! come with a clear upgrade path. To run speedyspeech I would have had to port the code to a much
//! newer torch version and retrained a model from scratch.
//!
//! This is generally a painful part of ML when libraries like
//! torch or tensorflow are updated there's no thought given to the upgrade flow for users. I think
//! this is caused by the view that once a piece of research is published it's often pushed to github and
//! abandoned in favour of the next project. Any changes that require an update in torch version
//! would likely be new research and justify retraining from scratch which removes the need to
//! upgrade version.
use crate::phonemes::*;
use std::str::FromStr;

pub mod speedy_ort;
pub use speedy_ort::*;

//pub mod speedy_tract;
//pub use speedy_tract::*;
//pub mod speedy_torch;
//pub use speedy_torch::*;
//pub mod speedy_candle;
//pub use speedy_candle::*;

// audio:
//  n_mel_channels: 80
//  segment_length: 16000
//  pad_short: 2000
//  filter_length: 1024
//  hop_length: 256 # WARNING: this can't be changed.
//  win_length: 1024
//  sampling_rate: 22050
//  mel_fmin: 0.0
//  mel_fmax: 8000.0

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
        Unit::Punct(Punctuation::FullStop),
        Unit::Punct(Punctuation::Comma),
        Unit::Punct(Punctuation::QuestionMark),
        Unit::Punct(Punctuation::ExclamationMark),
        Unit::Punct(Punctuation::Dash),
    ]);

    res
}
