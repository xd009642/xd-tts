use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_onnx::onnx::ModelProto;
use std::collections::HashMap;
use std::path::Path;

pub struct SpeedySpeech {
    model_proto: ModelProto,
}

impl SpeedySpeech {
    #[must_use]
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        // read all for debugging
        let model_proto = candle_onnx::read_file(path.as_ref()).unwrap();
        let graph = match model_proto.graph.as_ref() {
            None => anyhow::bail!("No graph included in ONNX"),
            Some(graph) => graph,
        };
        for input in &graph.input {
            println!("Graph input: {:?}", input);
        }
        for output in &graph.output {
            println!("Graph output: {:?}", output);
        }
        Ok(Self { model_proto })
    }

    pub fn infer(&self) -> anyhow::Result<()> {
        let inputs = HashMap::new();
        candle_onnx::simple_eval(&self.model_proto, inputs)?;
        todo!()
    }
}
