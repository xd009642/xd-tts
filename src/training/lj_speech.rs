use crate::text_normaliser::normalise_text;
use crate::CmuDictionary;
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::path::Path;
use tracing::{error, info};

pub struct Entry {
    pub id: String,
    pub text: String,
}

pub struct Dataset {
    pub entries: Vec<Entry>,
}

impl Dataset {
    pub fn load(p: impl AsRef<Path>) -> anyhow::Result<Self> {
        let f = File::open(p)?;
        let reader = io::BufReader::new(f);
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b'|')
            .quoting(false) // LJ004-0076 and others don't close quotes on first channel transcript...
            .flexible(true)
            .from_reader(reader);

        let mut entries = vec![];

        for result in rdr.records() {
            let record = result?;
            match (record.get(0), record.get(1)) {
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

    pub fn convert_to_pronunciation(&mut self, dict: &CmuDictionary) {
        for entry in self.entries.iter_mut() {
            todo!()
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
