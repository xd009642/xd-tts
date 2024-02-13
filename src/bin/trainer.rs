use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use tracing::info;
use xd_tts::training::*;
use xd_tts::*;

#[derive(Parser, Debug)]
pub struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Analyses the LJ Speech annotations and provides information on the corpus
    Analyse {
        /// LJ Speech metadata.csv
        #[clap(short, long, default_value = "./data/LJSpeech-1.1/metadata.csv")]
        input: PathBuf,
        /// Location to save the analysis json
        #[clap(short, long, default_value = "analysis.json")]
        output: PathBuf,
    },
    /// This prepares the data for training, for this I want to normalise the transcripts for LJ
    /// Speech, convert to the phonetic transcription (as per the tacotron2 text processing
    /// scripts) and write out a new set of training labels.
    Prepare {
        /// LJ Speech metadata.csv
        #[clap(short, long, default_value = "./data/LJSpeech-1.1/metadata.csv")]
        input: PathBuf,
        /// Location to save the fixed metadata csv
        #[clap(short, long, default_value = "phoneme_metadata.csv")]
        output: PathBuf,
    },
}

impl Commands {
    fn input(&self) -> &Path {
        match self {
            Self::Analyse { input, .. } => &input,
            Self::Prepare { input, .. } => &input,
        }
    }
}

fn main() -> anyhow::Result<()> {
    xd_tts::setup_logging();
    let args = Args::parse();
    let dictionary = CmuDictionary::open("./data/cmudict-0.7b.txt")?;
    info!("Dictionary size (words): {}", dictionary.len());

    let dataset = lj_speech::Dataset::load(args.command.input())?;

    match args.command {
        Commands::Analyse { output, .. } => {
            let mut analytics = AnalyticsGenerator::new(dictionary);

            for entry in dataset.entries.iter().map(|x| x.text.as_ref()) {
                analytics.push_sentence(entry);
            }
            let report = analytics.generate_report();

            info!("Number of OOV words: {}", report.oov.len());
            info!("Number of diphones: {}", report.diphones.len());
            info!("Number of phones: {}", report.phones.len());

            let report = serde_json::to_string_pretty(&report)?;
            std::fs::write(output, report)?;

            Ok(())
        }
        Commands::Prepare { output, .. } => {
            assert!(dataset.validate());
            todo!()
        }
    }
}
