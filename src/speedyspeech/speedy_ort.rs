use super::*;
use anyhow::Context;
use ndarray::{Array1, Array2, Axis};
use std::path::Path;
use ort::{inputs, GraphOptimizationLevel, Session};

pub struct SpeedyOrt {
    model: Session,
    phoneme_ids: Vec<Unit>,
}

impl SpeedyOrt {
    #[must_use]
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        // Load all the networks. Context is added to the error for nicer printouts
        // messes things up
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_file(path)
            .context("converting speedyspeech to runnable model")?;

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

        let plen = phonemes.len();

        let phonemes = Array2::<i64>::from_shape_vec((1, plen), phonemes)
            .context("failed to make phoneme tensor")?;

        let plen = Array1::from_vec(vec![plen as i64]);

        let inputs = inputs![
            "plen" => plen,
            "phonemes" => phonemes,
        ]?;

        // So torch can output invalid ONNX.
        //
        // Error: Failed to run inference on model: Non-zero status code returned while running Expand node. Name:'/Expand_8' Status Message: invalid expand shape

        let output = self.model.run(inputs)?;

        let spec = output["spec"]
            .extract_tensor::<f32>()?
            .view()
            .clone()
            .remove_axis(Axis(0))
            .into_dimensionality()?
            .into_owned();

        Ok(spec)
    }
}

