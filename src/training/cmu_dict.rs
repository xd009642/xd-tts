use crate::phonemes::*;
use crate::text_normaliser::*;
use std::collections::{btree_map, BTreeMap};
use std::fs;
use std::io::{self, prelude::*};
use std::path::Path;
use std::str::FromStr;
use tracing::error;

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

    pub fn merge(&mut self, other: CmuDictionary) {
        for (k, mut v) in other.dictionary.into_iter() {
            let pronunciations = self.dictionary.entry(k).or_default();
            for pronunc in v.drain(..) {
                if !pronunciations.contains(&pronunc) {
                    pronunciations.push(pronunc);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.dictionary.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
                Some(s) => dict_normalise(s),
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
                        error!("Unable to parse phone {}: {} for word: {}", i, e, word);
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
        self.get_pronunciations_normalised(&normalise_text(word).to_string_unchecked())
    }

    pub fn into_simple_dictionary(self) -> BTreeMap<String, Pronunciation> {
        self.dictionary
            .into_iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k, v[0].clone()))
            .collect()
    }

    pub fn iter(&self) -> btree_map::Iter<'_, String, Vec<Pronunciation>> {
        self.dictionary.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_merge() {
        let cursor = io::Cursor::new("RUSTNATION  R AH1 S T N EY1 SH AH0 N\nRUST  R AH1 S T");

        let mut base = CmuDictionary::from_reader(io::BufReader::new(cursor)).unwrap();

        let cursor = io::Cursor::new("RUSTNATION  R AH1 S T N EY1 SH AH0 N\nRUSTNATION  R AH1 S N EY1 SH AH0 N\nUST  UH1 S T");

        let to_merge = CmuDictionary::from_reader(io::BufReader::new(cursor)).unwrap();

        assert_eq!(base.len(), 2);
        assert_eq!(base.get_pronunciations("RUSTNATION").unwrap().len(), 1);
        assert_eq!(base.get_pronunciations("RUST").unwrap().len(), 1);
        assert_eq!(base.get_pronunciations("UST"), None);
        assert_eq!(to_merge.len(), 2);
        assert_eq!(to_merge.get_pronunciations("RUSTNATION").unwrap().len(), 2);
        assert_eq!(to_merge.get_pronunciations("RUST"), None);
        assert_eq!(to_merge.get_pronunciations("UST").unwrap().len(), 1);

        base.merge(to_merge);
        assert_eq!(base.len(), 3);
        assert_eq!(base.get_pronunciations("RUSTNATION").unwrap().len(), 2);
        assert_eq!(base.get_pronunciations("RUST").unwrap().len(), 1);
        assert_eq!(base.get_pronunciations("UST").unwrap().len(), 1);
    }
}
