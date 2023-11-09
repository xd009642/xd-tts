use crate::phonemes::*;
use crate::text_normaliser::*;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, prelude::*};
use std::path::Path;
use std::str::FromStr;

#[derive(Debug, Default, Clone)]
pub struct CmuDictionary {
    /// One word may have multiple pronunciations
    dictionary: BTreeMap<String, Vec<Pronunciation>>,
}

impl CmuDictionary {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        Self::from_reader(reader)
    }

    fn from_reader(reader: impl BufRead) -> io::Result<Self> {
        let mut dictionary: BTreeMap<String, Vec<Pronunciation>> = BTreeMap::new();

        'outer: for line in reader
            .lines()
            .filter_map(|x| x.ok())
            .filter(|x| !x.starts_with(";;;"))
        {
            let mut data = line.split("  ");
            let word = match data.next() {
                Some(s) => normalise_text(s),
                None => continue,
            };
            let phonemes = match data.next() {
                Some(s) => s.split(' '),
                None => continue,
            };
            let mut pronounce = vec![];
            for (i, phone) in phonemes
                .filter(|x| !x.is_empty())
                .map(|x| PhoneticUnit::from_str(x))
                .enumerate()
            {
                match phone {
                    Ok(s) => {
                        pronounce.push(s);
                    }
                    Err(e) => {
                        eprintln!("Unable to parse phone {}: {} for word: {}", i, e, word);
                        continue 'outer;
                    }
                }
            }
            match dictionary.get_mut(&word) {
                Some(s) => s.push(pronounce),
                None => {
                    dictionary.insert(word, vec![pronounce]);
                }
            }
        }
        Ok(Self { dictionary })
    }

    #[inline(always)]
    pub fn get_pronunciations_normalised(&self, word: &str) -> Option<&Vec<Pronunciation>> {
        self.dictionary.get(word)
    }

    pub fn get_pronunciations(&self, word: &str) -> Option<&Vec<Pronunciation>> {
        self.get_pronunciations_normalised(&normalise_text(word))
    }

    pub fn into_simple_dictionary(self) -> BTreeMap<String, Pronunciation> {
        self.dictionary
            .into_iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k, v[0].clone()))
            .collect()
    }
}
