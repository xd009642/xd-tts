use crate::phonemes::*;
use crate::text_normaliser::*;
use std::collections::BTreeMap;

pub struct G2pModel {
    dictionary: BTreeMap<String, Pronunciation>,
}

#[derive(Default)]
pub struct G2pModelBuilder {
    dict: Option<BTreeMap<String, Pronunciation>>,
}

impl G2pModelBuilder {
    pub fn add_dictionary(mut self, dict: BTreeMap<String, Pronunciation>) -> Self {
        self.dict = Some(dict);
        self
    }

    pub fn build(self) -> anyhow::Result<G2pModel> {
        match self.dict {
            Some(dict) => Ok(G2pModel { dictionary: dict }),
            None => anyhow::bail!("No means of working out pronunciation"),
        }
    }
}

impl G2pModel {
    pub fn create() -> G2pModelBuilder {
        G2pModelBuilder::default()
    }

    pub fn get_pronunciation(&self, word: &str) -> Option<&Pronunciation> {
        self.dictionary.get(&normalise_text(word))
    }

    pub fn get_pronunciation_normalised(&self, word: &str) -> Option<&Pronunciation> {
        self.dictionary.get(word)
    }
}
