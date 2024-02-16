//! A public domain single-speaker dataset with 24 hours often used in literature to train TTS
//! systems. It can be found [here](https://keithito.com/LJ-Speech-Dataset/). This file handles
//! loading it and a bit of fixing of the data.
//!
//! If you attempt to naively read the manifest you may encounter issues such as:
//!
//! 1. Unclosed quote marks in the CSV
//! 2. A different number of columns in some rows
//!
//! These sort of issues often exist in datasets and as such it's generally a good idea to write a
//! specific loader the moment you encounter any issues to make it convenient and act as a form of
//! documenting how you've changed the data. Ideally you don't want to apply fixes directly to a
//! dataset unless it's being versioned somewhere so you don't lose track of changes applied.
//!
//! We want to load this dataset and train a new tacotron2 model which has the phoneme inputs
//! trained and not producing gibberish!
use crate::phonemes::Unit;
use crate::text_normaliser::*;
use crate::CmuDictionary;
use csv::{ReaderBuilder, WriterBuilder};
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::path::Path;
use tracing::{debug, error, info};

/// An entry, a number of entries have two text fields one unnormalised and one partial normalised
/// (typically just numbers -> text).
pub struct Entry {
    /// ID of the entry
    pub id: String,
    /// A transcription of the utterance
    pub text: String,
}

/// Type containing the whole dataset
pub struct Dataset {
    /// List of entries
    pub entries: Vec<Entry>,
}

impl Dataset {
    /// Loads the lj speech manifest from a path
    pub fn load(p: impl AsRef<Path>) -> anyhow::Result<Self> {
        let f = File::open(p)?;
        let reader = io::BufReader::new(f);
        let mut rdr = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b'|')
            .quoting(false) // LJ004-0076 and others don't close quotes on first channel transcript...
            .flexible(true)
            .from_reader(reader);

        let mut entries = vec![];

        for result in rdr.records() {
            let record = result?;
            // So LJ Speech contains normalised transcripts as the 2nd field, we should prefer that
            // instead of normalising ourselves
            match (record.get(0), record.get(2).or_else(|| record.get(1))) {
                (Some(id), Some(text)) => {
                    assert!(!text.contains("|"), "Failed to split: {:?}", record);
                    entries.push(Entry {
                        id: id.to_string(),
                        text: text.to_string(),
                    });
                }
                _ => error!("Incomplete record: {:?}", record),
            }
        }
        Ok(Self { entries })
    }

    /// Write back our modified manifest with any changes we've applied to the transcripts.
    pub fn write_csv(&self, writer: impl io::Write) -> anyhow::Result<()> {
        let mut writer = WriterBuilder::new()
            .has_headers(false)
            .delimiter(b'|')
            .flexible(true)
            .from_writer(writer);

        for entry in &self.entries {
            writer.write_record(&[entry.id.as_str(), entry.text.as_str(), entry.text.as_str()])?;
        }
        Ok(())
    }

    /// Converts words to their phonetic representations. This will generally work more reliably if
    /// the transcripts are already normalised. But we do run our text normaliser and attempt to
    /// normalise anything that isn't already normalised.
    pub fn convert_to_pronunciation(&mut self, dict: &CmuDictionary) {
        for entry in self.entries.iter_mut() {
            let mut normalised = normalise_text(&entry.text);
            normalised.words_to_pronunciation(dict);
            let mut new_string = String::new();
            for chunk in normalised.drain_all() {
                match chunk {
                    NormaliserChunk::Pronunciation(units) if !units.is_empty() => {
                        let mut tmp = String::new();
                        let mut in_pronunciation = false;
                        for unit in units.iter() {
                            match unit {
                                Unit::Phone(p) => {
                                    if !in_pronunciation {
                                        tmp.push('{');
                                        in_pronunciation = true;
                                    }
                                    tmp.push_str(p.to_string().as_str());
                                    tmp.push(' ');
                                }
                                Unit::Space => {
                                    if in_pronunciation {
                                        tmp.push('}');
                                    }
                                    in_pronunciation = false;
                                    tmp.push(' ');
                                }
                                Unit::Punct(p) => {
                                    if in_pronunciation {
                                        tmp.push('}');
                                    }
                                    in_pronunciation = false;
                                    tmp.push_str(p.to_string().as_str());
                                    tmp.push(' ');
                                }
                                e => panic!("Unexpected unit: {:?}", e),
                            }
                        }
                        new_string.push_str(tmp.as_str());
                    }
                    NormaliserChunk::Punct(p) => {
                        new_string.push_str(p.to_string().as_str());
                        new_string.push(' ');
                    }
                    NormaliserChunk::Pronunciation(_) => {}
                    e => {
                        panic!("Didn't expect: {:?}", e);
                    }
                }
            }
            debug!("Replacing string!");
            debug!("Old string: {}", entry.text);
            debug!("New string: {}", new_string);
            entry.text = new_string;
        }
    }

    /// Validates there's nothing wrong with the dataset. Will log any errors it finds and return
    /// false
    pub fn validate(&self) -> bool {
        info!("Validating dataset");
        let mut ids = HashSet::new();
        let mut success = true;
        for entry in &self.entries {
            if entry.text.trim().is_empty() {
                error!("Transcript for {} is empty", entry.id);
                success = false;
            }
            let normalised = normalise_text(&entry.text).to_string();
            match normalised {
                Ok(s) if s.trim().is_empty() => {
                    error!(
                        "{} transcript '{}' normalises to an empty string",
                        entry.id, entry.text
                    );
                    success = false;
                }
                Err(e) => {
                    error!(
                        "{} failed to generate string from normaliser output: {}",
                        entry.id, e
                    );
                    success = false;
                }
                Ok(_) => {}
            }
            if ids.contains(entry.id.as_str()) {
                error!("Duplicate ID: {}", entry.id);
                success = false;
            }
            ids.insert(entry.id.as_str());
        }
        info!("Validation complete");
        success
    }
}
