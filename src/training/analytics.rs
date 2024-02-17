//! Does some analytics on datasets, this module is largely a result from initial thoughts into
//! other models and more involved training where understanding some more parts of the dataset and
//! language could be helpful for modelling. I've preserved it here because it doesn't take
//! anything away, but it is less relevant for systems like ours utilising a neural network.
use crate::phonemes::*;
use crate::text_normaliser::*;
use crate::CmuDictionary;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tracing::info;

/// A diphone is a neighbouring pair of phones, and a phone is a distinct speech sound. A phone
/// differs from a phoneme because if you change the phoneme you change the word and it's
/// meaning but phones could potentially be switched and it would be more akin to changing the
/// pronunciation but not meaning of the word. Therefore a language will have more possible
/// phones than phonemes. If a language has P phones.
///
/// If we have P phones in a language the maximum possible diphones is P^2. However, languages
/// typically have rules on what sounds can occur near each other so in practice the number is a
/// lot lower than P^2. Some synthesis techniques will use diphones because it can allow for a
/// sliding window for synthesis and easier blending with less visible cuts.
///
/// We do use phonemes for our diphones from the dataset so there is an inaccuracy of sorts in the
/// analytics, but whether that is acceptable or a negative would depend on what models/approaches
/// it went into.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct DiphoneStat {
    /// The phone pair
    pub phones: [String; 2],
    /// The count in the dataset
    pub count: usize,
}

/// The end analytics from the dataset, this is just designed to be serialized out to a json or
/// similar structure. Though all fields are public to enable it to be used in code.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Analytics {
    /// List of diphones and counts
    pub diphones: Vec<DiphoneStat>,
    /// List of phonemes and their counts
    pub phonemes: BTreeMap<String, usize>,
    /// Out of vocabulary words
    pub oov: BTreeMap<String, usize>,
    /// Lengths of sentences being synthesised. This is generally good to get an idea of the
    /// longest context you're seeing during training as things beyond this may end up posing
    /// issues.
    pub sentence_lengths: BTreeMap<usize, usize>,
}

/// Used to generate analytics, this is because some of the running state may not want to be
/// serialized or may otherwise be unserialisable (taking json as a target format).
#[derive(Debug, Default)]
pub struct AnalyticsGenerator {
    /// Dictionary used for anaytics
    dict: CmuDictionary,
    /// Map to keep track of the diphones
    diphones: BTreeMap<[PhoneticUnit; 2], usize>,
    /// Running count of phonemes
    phonemes: BTreeMap<PhoneticUnit, usize>,
    /// Running count of OOVs
    oov: BTreeMap<String, usize>,
    /// Running count of sentence lengths
    sentence_lengths: BTreeMap<usize, usize>,
}

impl AnalyticsGenerator {
    /// Create a new analytics generator
    pub fn new(dict: CmuDictionary) -> Self {
        Self {
            dict,
            ..Default::default()
        }
    }

    /// Adds the word into the analysis
    pub fn push_word(&mut self, word: &str) {
        let normalised = normalise_text(word).to_string_unchecked();
        if let Some(pronunciations) = self.dict.get_pronunciations_normalised(&normalised) {
            for pronunciation in pronunciations.iter() {
                for window in pronunciation.as_slice().windows(2) {
                    *self.diphones.entry([window[0], window[1]]).or_insert(0) += 1;
                    *self.phonemes.entry(window[0]).or_insert(0) += 1;
                }
                // This will skip adding the last one to the phones map so do it here
                if let Some(last) = pronunciation.last() {
                    *self.phonemes.entry(*last).or_insert(0) += 1;
                }
            }
        } else {
            *self.oov.entry(normalised).or_insert(0) += 1;
        }
    }

    /// Process a sentence and also add all the words in it into the analysis
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

    /// Generates a report, this can be saved as a json for future processing.
    pub fn generate_report(&self) -> Analytics {
        let diphones = self
            .diphones
            .iter()
            .map(|(k, v)| DiphoneStat {
                phones: [k[0].to_string(), k[1].to_string()],
                count: *v,
            })
            .collect();

        let phonemes = self
            .phonemes
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        Analytics {
            diphones,
            phonemes,
            oov: self.oov.clone(),
            sentence_lengths: self.sentence_lengths.clone(),
        }
    }
}
