use super::*;
use anyhow::Context;
use ndarray::Array2;
use std::path::Path;
use ort::{tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Session,
	Value
};
use tracing::info;
use ndarray::prelude::*;


pub struct SpeedyOrt {
    model: Session,
    phoneme_ids: Vec<Unit>,
}

impl SpeedyOrt {
    #[must_use]
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {

        let environment = Environment::builder()
            .with_name("xd_tts")
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .build()?
            .into_arc();


        let model = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_model_from_file(path)
            .context("converting to runnable model")?;

        info!("{:?}", model);

        Ok(Self {
            model,
            phoneme_ids: generate_id_list(),
        })
    }

    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        let mut phonemes = units
            .iter()
            .map(|x| best_match_for_unit(x, &self.phoneme_ids))
            .collect::<Vec<_>>();

        phonemes.resize(phonemes.len() + 5, 0);

        info!("Phonemes: {:?}", phonemes);

        let plen = CowArray::from(arr1(&[phonemes.len() as i64])).into_dyn();

        let phonemes = Array2::from_shape_vec((1, phonemes.len()), phonemes).context("invalid dimensions")?;
        let phonemes = CowArray::from(phonemes).into_dyn();

        let inputs = vec![Value::from_array(self.model.allocator(), &phonemes)?, Value::from_array(self.model.allocator(), &plen)?];

        let session_outputs = self.model.run(inputs)?;

        todo!();

        //let tensor = Tensor::from_shape(&[1, units.len()], &phonemes)?;
        //let plen = Tensor::from(units.len() as i64);

        //let result = self.model.run(tvec!(tensor.into(), plen.into()))?;

        todo!()
    }
}
