use clap::Parser;
use griffin_lim::GriffinLim;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::{error, info};
use xd_tts::phonemes::Unit;
use xd_tts::tacotron2::*;
use xd_tts::text_normaliser::{self, NormaliserChunk};
use xd_tts::training::cmu_dict::*;

#[derive(Parser, Debug)]
pub struct Args {
    /// Text to synthesise speech for
    #[clap(long, short)]
    input: String,
    /// Saves the generated spectrograms for debugging purposes
    #[clap(long)]
    output_spectrogram: Option<PathBuf>,
    /// Location to save the output audio file
    #[clap(short, long, default_value = "output.wav")]
    output: PathBuf,
    /// If true characters and input into tacotron2, otherwise the phoneme inputs are used
    #[clap(long)]
    phoneme_input: bool,
    /// Directory where the tacotron2 ONNX models can be found
    #[clap(long, default_value = "./models/tacotron2")]
    tacotron2: PathBuf,
}

fn create_wav_writer(output: &Path) -> anyhow::Result<WavWriter<BufWriter<File>>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 22050,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let w = WavWriter::create(output, spec)?;
    Ok(w)
}

fn write_silence(
    duration: Duration,
    wav_writer: &mut WavWriter<BufWriter<File>>,
) -> anyhow::Result<()> {
    let n_samples = (wav_writer.spec().sample_rate as f32 * duration.as_secs_f32()).round() as u32;

    if n_samples > 0 {
        let mut i16_writer = wav_writer.get_i16_writer(n_samples);
        for _ in 0..n_samples {
            i16_writer.write_sample(0);
        }
        i16_writer.flush()?;
    }
    Ok(())
}

fn infer(
    args: &Args,
    input: &[Unit],
    model: &Tacotron2,
    vocoder: &GriffinLim,
    wav_writer: &mut WavWriter<BufWriter<File>>,
) -> anyhow::Result<()> {
    if input.is_empty() {
        return Ok(());
    }
    let mel_gen_start = Instant::now();
    let spectrogram = model.infer(input)?;

    if let Some(output_spectrogram) = args.output_spectrogram.as_ref() {
        // use wav_writer.duration() to add start_sample
        let output_spectrogram = if wav_writer.duration() > 0 {
            todo!()
        } else {
            output_spectrogram.clone()
        };
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
    info!("Mel gen time: {:?}", vocoder_start - mel_gen_start);
    info!("Vocoder time: {:?}", end - vocoder_start);
    info!(
        "Real time factor: {}",
        (end - mel_gen_start).as_secs_f32() / audio_length
    );

    let mut i16_writer = wav_writer.get_i16_writer(audio.len() as u32);
    for sample in &audio {
        i16_writer.write_sample((*sample * i16::MAX as f32) as i16);
    }
    i16_writer.flush()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    xd_tts::setup_logging();
    let args = Args::parse();

    info!("Loading resources");

    let mut dict = CmuDictionary::open("data/cmudict-0.7b.txt")?;
    if let Ok(custom) = CmuDictionary::open("resources/custom_dict.txt") {
        dict.merge(custom);
    }
    let model = Tacotron2::load(&args.tacotron2)?;
    let vocoder = create_griffin_lim()?;
    //let model = SpeedyTorch::load("./models/model_file.pth")?;

    let start = Instant::now();
    info!("Text normalisation");
    let mut text = text_normaliser::normalise(&args.input)?;
    if args.phoneme_input {
        // Sad tacotron2 was trained with ARPA support
        text.words_to_pronunciation(&dict);
    } else {
        text.convert_to_units();
    }
    let mut inference_chunk = vec![];
    let mut wav_writer = create_wav_writer(&args.output)?;

    let text_end = Instant::now();
    info!("Text processing time: {:?}", text_end - start);
    info!("Generating audio");
    for chunk in text.drain_all() {
        match chunk {
            NormaliserChunk::Pronunciation(mut units) => inference_chunk.append(&mut units),
            NormaliserChunk::Break(duration) => {
                // Infer here.
                // Potentially we could use the alignments in the network output and return them
                // with the spectrogram to insert this stuff. That might be better - it depends if
                // coarticulation sounds more or less natural when a giant pause is inserted.
                let _spectrogram = model.infer(&inference_chunk)?;

                infer(&args, &inference_chunk, &model, &vocoder, &mut wav_writer)?;
                write_silence(duration, &mut wav_writer)?;
                inference_chunk.clear();
            }
            NormaliserChunk::Text(t) => {
                unreachable!("'{}' Should have been converted to pronunciation", t)
            }
            NormaliserChunk::Punct(p) => {
                inference_chunk.push(Unit::Punct(p));
            }
        }
    }
    infer(&args, &inference_chunk, &model, &vocoder, &mut wav_writer)?;
    let end = Instant::now();
    info!("Finished processing in: {:?}", end - start);
    Ok(())
}
