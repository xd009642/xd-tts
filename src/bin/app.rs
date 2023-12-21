use clap::Parser;
use tracing::{info, warn};
use xd_tts::phonemes::Unit;
use xd_tts::tacotron2::*;
use xd_tts::text_normaliser::{self, NormaliserChunk};
use xd_tts::training::cmu_dict::*;

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(long, short)]
    input: String,
}

fn main() -> anyhow::Result<()> {
    xd_tts::setup_logging();
    let args = Args::parse();

    info!("Loading resources");

    let mut dict = CmuDictionary::open("data/cmudict-0.7b.txt")?;
    if let Ok(custom) = CmuDictionary::open("resources/custom_dict.txt") {
        dict.merge(custom);
    }
    let model = Tacotron2::load("./models/tacotron2")?;
    //let model = SpeedyTorch::load("./models/model_file.pth")?;

    info!("Text normalisation");
    let mut text = text_normaliser::normalise(&args.input)?;
    text.words_to_pronunciation(&dict);

    let mut inference_chunk = vec![];

    info!("Generating audio");
    for chunk in text.drain_all() {
        match chunk {
            NormaliserChunk::Pronunciation(mut units) => inference_chunk.append(&mut units),
            NormaliserChunk::Break(_duration) => {
                // Infer here.
                // Potentially we could use the alignments in the network output and return them
                // with the spectrogram to insert this stuff. That might be better - it depends if
                // coarticulation sounds more or less natural when a giant pause is inserted.
                let _spectrogram = model.infer(&inference_chunk)?;
                inference_chunk.clear();

                warn!("How do I break!?");
            }
            NormaliserChunk::Text(t) => {
                unreachable!("'{}' Should have been converted to pronunciation", t)
            }
            NormaliserChunk::Punct(p) => {
                inference_chunk.push(Unit::Punct(p));
            }
        }
    }
    if !inference_chunk.is_empty() {
        let _spectrogram = model.infer(&inference_chunk)?;
    }
    Ok(())
}
