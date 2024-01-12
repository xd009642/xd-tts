//! Does some analytics on datasets.
use super::*;
use crate::phonemes::*;
use crate::text_normaliser::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tracing::info;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct DiphoneStat {
    pub phones: [String; 2],
    pub count: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Analytics {
    pub diphones: Vec<DiphoneStat>,
    pub phones: BTreeMap<String, usize>,
    /// Out of vocabulary words
    pub oov: BTreeMap<String, usize>,
    pub sentence_lengths: BTreeMap<usize, usize>,
}

#[derive(Debug, Default)]
pub struct AnalyticsGenerator {
    /// Dictionary used for anaytics
    dict: CmuDictionary,
    diphones: BTreeMap<[PhoneticUnit; 2], usize>,
    phones: BTreeMap<PhoneticUnit, usize>,
    oov: BTreeMap<String, usize>,
    sentence_lengths: BTreeMap<usize, usize>,
}

impl AnalyticsGenerator {
    pub fn new(dict: CmuDictionary) -> Self {
        Self {
            dict,
            ..Default::default()
        }
    }

    pub fn push_word(&mut self, word: &str) {
        let normalised = normalise_text(word).to_string_unchecked();
        if let Some(pronunciations) = self.dict.get_pronunciations_normalised(&normalised) {
            for pronunciation in pronunciations.iter() {
                for window in pronunciation.as_slice().windows(2) {
                    *self.diphones.entry([window[0], window[1]]).or_insert(0) += 1;
                    *self.phones.entry(window[0]).or_insert(0) += 1;
                }
                // This will skip adding the last one to the phones map so do it here
                if let Some(last) = pronunciation.last() {
                    *self.phones.entry(*last).or_insert(0) += 1;
                }
            }
        } else {
            *self.oov.entry(normalised).or_insert(0) += 1;
        }
    }

    pub fn push_sentence(&mut self, sentence: &str) {
        let mut text = normalise(sentence).unwrap();
        text.words_to_pronunciation(&self.dict);
        let mut sentence_len = 0;
        for chunk in text.drain_all() {
            match chunk {
                NormaliserChunk::Pronunciation(units) => {
                    sentence_len += units.len();
                }
                NormaliserChunk::Text(t) => {
                    unreachable!("'{}' Should have been converted to pronunciation", t)
                }
                NormaliserChunk::Break(_duration) => {}
                NormaliserChunk::Punct(p) => {
                    sentence_len += 1;
                    if p.is_sentence_end() {
                        *self.sentence_lengths.entry(sentence_len).or_default() += 1;
                        if sentence_len > 160 {
                            info!("Very long sentence found: '{}'", sentence);
                        }
                        sentence_len = 0;
                    }
                }
            }
        }
        if sentence_len > 0 {
            *self.sentence_lengths.entry(sentence_len).or_default() += 1;
            if sentence_len > 160 {
                info!("Very long sentence found: '{}'", sentence);
            }
        }
        for word in sentence.split_whitespace() {
            self.push_word(word);
        }
    }

    pub fn generate_report(&self) -> Analytics {
        let diphones = self
            .diphones
            .iter()
            .map(|(k, v)| DiphoneStat {
                phones: [k[0].to_string(), k[1].to_string()],
                count: *v,
            })
            .collect();

        let phones = self
            .phones
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        Analytics {
            diphones,
            phones,
            oov: self.oov.clone(),
            sentence_lengths: self.sentence_lengths.clone(),
        }
    }
}
