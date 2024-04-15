use clap::Parser;
use griffin_lim::GriffinLim;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::{debug, error, info};
use xd_tts::phonemes::Unit;
use xd_tts::tacotron2::*;
use xd_tts::text_normaliser::{self, NormaliserChunk};
use xd_tts::*;

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
    /// If set phonemes and input into tacotron2, by default character inputs are used
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

fn main() -> anyhow::Result<()> {
    xd_tts::setup_logging();
    let args = Args::parse();

    info!("Loading resources");

    let tts_context = XdTts::new(&args.tacotron2, args.phoneme_input)?;
    let mut wav_writer = create_wav_writer(&args.output)?;

    tts_context.generate_audio(&args.input, &mut wav_writer, args.output_spectrogram)?;
    Ok(())
}
