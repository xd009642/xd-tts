use super::*;
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

pub struct SpeedyCandle {
    model_proto: ModelProto,
    phoneme_ids: Vec<Unit>,
}

impl SpeedyCandle {
    #[must_use]
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        // read all for debugging
        let model_proto = candle_onnx::read_file(path.as_ref()).unwrap();
        let graph = match model_proto.graph.as_ref() {
            None => anyhow::bail!("No graph included in ONNX"),
            Some(graph) => graph,
        };
        for input in &graph.input {
            info!("Graph input: {:?}", input);
        }
        for output in &graph.output {
            info!("Graph output: {:?}", output);
        }
        Ok(Self {
            model_proto,
            phoneme_ids: generate_id_list(),
        })
    }

    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        let graph = self.model_proto.graph.as_ref().unwrap();

        let mut inputs = HashMap::new();

        for input in graph.input.iter() {
            let value = if input.name == "phonemes" {
                // Phonemes is a sequence tensor of [batch_size, phonemes]
                let phonemes = units
                    .iter()
                    .map(|x| best_match_for_unit(x, &self.phoneme_ids).unwrap_or(2))
                    .collect::<Vec<_>>();
                Tensor::from_vec(phonemes, (1, units.len()), &Device::Cpu)?
            } else if input.name == "plen" {
                Tensor::from_iter([units.len() as i64], &Device::Cpu)?
            } else {
                anyhow::bail!("Unexpected input: {:?}", input);
            };
            inputs.insert(input.name.clone(), value);
        }

        let result = candle_onnx::simple_eval(&self.model_proto, inputs)?;
        // So lets just get rid of the phoneme durations since I don't care for them
        if let Some(spectrogram) = result.get("spec") {
            let shape = spectrogram.dims();
            let data = spectrogram.to_vec1::<f32>()?;
            // Outer most dimension is the batch size which is always 1 so we discard it.
            // We need to figure out ifg the norm inverse is required or included in the graph!
            Ok(Array2::from_shape_vec((shape[1], shape[2]), data)?)
        } else {
            anyhow::bail!("No spectrogram provided on output!");
        }
    }
}
