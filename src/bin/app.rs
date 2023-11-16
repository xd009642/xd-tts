use clap::Parser;
use xd_tts::phonemes::*;
use xd_tts::speedy_candle::*;
use xd_tts::training::cmu_dict::*;
use tracing::{info, warn};

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(long, short)]
    input: String,
}

fn main() -> anyhow::Result<()> {
    xd_tts::setup_logging();
    let args = Args::parse();

    let mut dict = CmuDictionary::open("data/cmudict-0.7b.txt")?;
    if let Ok(custom) = CmuDictionary::open("resources/custom_dict.txt") {
        dict.merge(custom);
    }
    let model = SpeedySpeech::load("./models/speedyspeech.onnx")?;

    let mut words = vec![];

    for word in args.input.split_whitespace() {
        if let Some(pronunciation) = dict.get_pronunciations(word) {
            assert!(!pronunciation.is_empty());
            info!("{} is pronounced: {:?}", word, pronunciation);
            words.extend(pronunciation[0].iter().map(|x| Unit::Phone(*x)));
            words.push(Unit::Space);
        } else {
            warn!("Unsupported word: '{}'", word);
        }
    }

    let _spectrogram = model.infer(&words)?;

    Ok(())
}
