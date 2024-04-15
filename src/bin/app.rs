use clap::Parser;
use hound::WavWriter;
use std::path::PathBuf;
use tracing::info;
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

fn main() -> anyhow::Result<()> {
    xd_tts::setup_logging();
    let args = Args::parse();

    info!("Loading resources");

    let tts_context = XdTts::new(&args.tacotron2, args.phoneme_input)?;
    let mut wav_writer = WavWriter::create(&args.output, xd_tts::WAV_SPEC)?;

    tts_context.generate_audio(&args.input, &mut wav_writer, args.output_spectrogram)?;
    Ok(())
}
