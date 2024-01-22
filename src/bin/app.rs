use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;
use tracing::{error, info, warn};
use xd_tts::phonemes::{Punctuation, Unit};
use xd_tts::tacotron2::*;
use xd_tts::text_normaliser::{self, NormaliserChunk};
use xd_tts::training::cmu_dict::*;

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(long, short)]
    input: String,
    #[clap(long)]
    output_spectrogram: Option<PathBuf>,
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
    let vocoder = create_griffin_lim()?;
    //let model = SpeedyTorch::load("./models/model_file.pth")?;

    let start = Instant::now();
    info!("Text normalisation");
    let mut text = text_normaliser::normalise(&args.input)?;
    // Sad tacotron2 was trained with ARPA support
    //text.words_to_pronunciation(&dict);
    text.convert_to_units();
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
        info!("Running {} tokens through mel gen", inference_chunk.len());
        let mel_gen_start = Instant::now();
        let spectrogram = model.infer(&inference_chunk)?;

        if let Some(output_spectrogram) = args.output_spectrogram {
            if let Err(e) = ndarray_npy::write_npy(&output_spectrogram, &spectrogram) {
                error!(
                    "Failed to write spectrogram to '{}': {}",
                    output_spectrogram.display(),
                    e
                );
            }
        }
        let vocoder_start = Instant::now();
        let audio = vocoder.infer(&spectrogram)?;

        let end = Instant::now();

        let audio_length = audio.len() as f32 / 22050.0;
        info!("Generated {}s of audio in {:?}", audio_length, end - start);
        info!("Text processing time: {:?}", mel_gen_start - start);
        info!("Mel gen time: {:?}", vocoder_start - mel_gen_start);
        info!("Vocoder time: {:?}", end - vocoder_start);

        let spec = WavSpec {
            channels: 1,
            sample_rate: 22050,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut wav_writer = WavWriter::create("output.wav", spec)?;

        let mut i16_writer = wav_writer.get_i16_writer(audio.len() as u32);
        for sample in &audio {
            i16_writer.write_sample((*sample * i16::MAX as f32) as i16);
        }
        i16_writer.flush()?;
    }
    Ok(())
}
