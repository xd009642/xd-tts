use super::*;
use anyhow::Context;
use ndarray::Array2;
use std::path::Path;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::InferenceOp;

pub struct SpeedyTract {
    model:
        SimplePlan<InferenceFact, Box<dyn InferenceOp>, Graph<InferenceFact, Box<dyn InferenceOp>>>,
    phoneme_ids: Vec<Unit>,
}

impl SpeedyTract {
    #[must_use]
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .context("loading ONNX file")?
            // https://github.com/sonos/tract/issues/1263
            //    .into_optimized()
            //    .context("optimising graph")?
            .into_runnable()
            .context("converting to runnable model")?;

        Ok(Self {
            model,
            phoneme_ids: generate_id_list(),
        })
    }

    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        let phonemes = units
            .iter()
            .map(|x| best_match_for_unit(x, &self.phoneme_ids).unwrap_or(2))
            .collect::<Vec<_>>();

        let tensor = Tensor::from_shape(&[1, units.len()], &phonemes)?;
        let plen = Tensor::from(units.len() as i64);

        let result = self.model.run(tvec!(tensor.into(), plen.into()))?;

        tracing::info!("Result: {:?}", result);

        todo!()
    }
}
