use super::*;
use burn::backend::ndarray::NdArray;
use ndarray::Array2;
use std::path::Path;

pub mod speedy {
    include!(concat!(env!("OUT_DIR"), "/model/speedyspeech.rs"));
}

pub struct SpeedyBurn {
    model: speedy::Model<NdArray>,
}

impl SpeedyBurn {
    #[must_use]
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model = speedy::Model::load_state();
        Ok(Self { session })
    }

    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        todo!();
    }
}
