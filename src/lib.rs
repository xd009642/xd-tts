#![doc = include_str!("../README.md")]
use crate::phonemes::Unit;
use crate::tacotron2::*;
use crate::text_normaliser::NormaliserChunk;
use griffin_lim::GriffinLim;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::env;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::{debug, error, info};
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::{Layer, Registry};

pub mod cmu_dict;
pub mod phonemes;
// This failed for various reasons. Look in the module so see the pains of ML.
//pub mod speedyspeech;
pub mod tacotron2;
pub mod text_normaliser;
pub mod training;

pub use cmu_dict::CmuDictionary;

pub const WAV_SPEC: WavSpec = WavSpec {
    channels: 1,
    sample_rate: 22050,
    bits_per_sample: 16,
    sample_format: SampleFormat::Int,
};

pub struct XdTts {
    dict: CmuDictionary,
    model: Tacotron2,
    vocoder: GriffinLim,
    phoneme_input: bool,
}

impl XdTts {
    pub fn new(tacotron2: &Path, phoneme_input: bool) -> anyhow::Result<Self> {
        let dict = if phoneme_input {
            let mut dict = CmuDictionary::open("data/cmudict-0.7b.txt")?;
            if let Ok(custom) = CmuDictionary::open("resources/custom_dict.txt") {
                dict.merge(custom);
            }
            dict
        } else {
            CmuDictionary::default()
        };
        let model = Tacotron2::load(tacotron2)?;
        let vocoder = create_griffin_lim()?;
        Ok(Self {
            dict,
            model,
            vocoder,
            phoneme_input,
        })
    }

    pub fn generate_audio<W>(
        &self,
        text: &str,
        wav_writer: &mut WavWriter<W>,
        output_spectrogram: Option<PathBuf>,
    ) -> anyhow::Result<()>
    where
        W: Write + Seek,
    {
        let start = Instant::now();
        info!("Text normalisation");
        let mut text = text_normaliser::normalise(text)?;
        if self.phoneme_input {
            // Sad tacotron2 was trained with ARPA support
            text.words_to_pronunciation(&self.dict);
        } else {
            text.convert_to_units();
        }
        let mut inference_chunk = vec![];

        let text_end = Instant::now();
        info!("Text processing time: {:?}", text_end - start);
        info!("Generating audio");
        for chunk in text.drain_all() {
            debug!("Chunk: {:?}", chunk);
            match chunk {
                NormaliserChunk::Pronunciation(mut units) => inference_chunk.append(&mut units),
                NormaliserChunk::Break(duration) => {
                    // Infer here.
                    // Potentially we could use the alignments in the network output and return them
                    // with the spectrogram to insert this stuff. That might be better - it depends if
                    // coarticulation sounds more or less natural when a giant pause is inserted.
                    self.infer(&inference_chunk, wav_writer, output_spectrogram.as_ref())?;
                    write_silence(duration, wav_writer)?;
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
        self.infer(&inference_chunk, wav_writer, output_spectrogram.as_ref())?;
        let end = Instant::now();
        info!("Finished processing in: {:?}", end - start);
        Ok(())
    }

    fn infer<W>(
        &self,
        input: &[Unit],
        wav_writer: &mut WavWriter<W>,
        output_spectrogram: Option<&PathBuf>,
    ) -> anyhow::Result<()>
    where
        W: Write + Seek,
    {
        if input.is_empty() {
            return Ok(());
        }
        let mel_gen_start = Instant::now();
        let spectrogram = self.model.infer(input)?;

        if let Some(output_spectrogram) = output_spectrogram {
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
        let audio = self.vocoder.infer(&spectrogram)?;

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
}

fn write_silence<W>(duration: Duration, wav_writer: &mut WavWriter<W>) -> anyhow::Result<()>
where
    W: Write + Seek,
{
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

/// Convenience function to setup logging for any binaries I create. Automatically sets all
/// binaries and the tts library crate to `info` logging by default.
pub fn setup_logging() {
    let filter = match env::var("RUST_LOG") {
        Ok(_) => EnvFilter::from_env("RUST_LOG"),
        _ => EnvFilter::new("xd_tts=info,app=info,trainer=info"),
    };

    let fmt = tracing_subscriber::fmt::Layer::default();

    let subscriber = filter.and_then(fmt).with_subscriber(Registry::default());

    tracing::subscriber::set_global_default(subscriber).unwrap();
}
