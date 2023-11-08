use clap::Parser;
use xd_tts::phonemes::*;
use xd_tts::speedy_candle::*;
use xd_tts::training::cmu_dict::*;

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(long, short)]
    input: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let dict = CmuDictionary::open("data/cmudict-0.7b.txt")?;
    let model = SpeedySpeech::load("./models/speedyspeech.onnx")?;

    let mut words = vec![];

    for word in args.input.split_whitespace() {
        if let Some(pronunciation) = dict.get_pronunciations(word) {
            assert!(!pronunciation.is_empty());
            words.extend(pronunciation[0].iter().map(|x| Unit::Phone(*x)));
            words.push(Unit::Space);
        } else {
            println!("Unsupported word: '{}'", word);
        }
    }

    let spectrogram = model.infer(&words)?;

    Ok(())
}
