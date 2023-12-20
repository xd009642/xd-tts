use crate::phonemes::*;
use anyhow::Context;
use ndarray::prelude::*;
use ndarray::Array2;
use ort::{
    tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult,
    Session, SessionBuilder, Value,
};
use std::path::Path;
use std::str::FromStr;
use tracing::info;

fn generate_id_list() -> Vec<Unit> {
    let phones = [
        "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2", "AO",
        "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2", "B", "CH", "D",
        "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1", "EY2", "F",
        "G", "HH", "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N",
        "NG", "OW", "OW0", "OW1", "OW2", "OY", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH",
        "UH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
    ];

    let mut res = vec![
        Unit::Padding,
        Unit::Punct(Punctuation::Dash),
        Unit::Punct(Punctuation::ExclamationMark),
        Unit::Punct(Punctuation::Apostrophe),
        Unit::Punct(Punctuation::OpenBracket),
        Unit::Punct(Punctuation::CloseBracket),
        Unit::Punct(Punctuation::Comma),
        Unit::Punct(Punctuation::FullStop),
        Unit::Punct(Punctuation::Colon),
        Unit::Punct(Punctuation::SemiColon),
        Unit::Punct(Punctuation::QuestionMark),
        Unit::Space,
    ];
    let characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        .chars()
        .map(|x| Unit::Character(x));

    res.extend(characters);
    res.extend(phones.iter().map(|x| Unit::from_str(x).unwrap()));

    res
}

// https://catalog.ngc.nvidia.com/orgs/nvidia/models/tacotron2pyt_jit_fp16/files

pub struct Tacotron2 {
    encoder: Session,
    decoder: Session,
    post_net: Session,
    phoneme_ids: Vec<Unit>,
}

struct DecoderState<'a> {
    decoder_input: CowArray<'a, f32, IxDyn>,
    attention_hidden: CowArray<'a, f32, IxDyn>,
    attention_cell: CowArray<'a, f32, IxDyn>,
    decoder_hidden: CowArray<'a, f32, IxDyn>,
    decoder_cell: CowArray<'a, f32, IxDyn>,
    attention_weights: CowArray<'a, f32, IxDyn>,
    attention_weights_cum: CowArray<'a, f32, IxDyn>,
    attention_context: CowArray<'a, f32, IxDyn>,
    //    memory: CowArray<f32, Ix3>,
    //    processed_memory: CowArray<f32, Ix3>,
    mask: CowArray<'a, bool, IxDyn>,
}

impl<'a> DecoderState<'a> {
    fn new(
        memory: &ArrayViewD<'a, f32>,
        processed_memory: &ArrayViewD<'a, f32>,
        memory_lengths: &ArrayViewD<'a, i64>,
    ) -> Self {
        let bs = memory.shape()[0];
        let seq_len = memory.shape()[1];
        let attention_rnn_dim = 1024;
        let decoder_rnn_dim = 1024;
        let encoder_embedding_dim = 512;
        let n_mel_channels = 80;

        let attention_hidden = CowArray::from(Array2::zeros((bs, attention_rnn_dim))).into_dyn();
        let attention_cell = CowArray::from(Array2::zeros((bs, attention_rnn_dim))).into_dyn();
        let decoder_hidden = CowArray::from(Array2::zeros((bs, decoder_rnn_dim))).into_dyn();
        let decoder_cell = CowArray::from(Array2::zeros((bs, decoder_rnn_dim))).into_dyn();
        let attention_weights = CowArray::from(Array2::zeros((bs, seq_len))).into_dyn();
        let attention_weights_cum = CowArray::from(Array2::zeros((bs, seq_len))).into_dyn();
        let attention_context =
            CowArray::from(Array2::zeros((bs, encoder_embedding_dim))).into_dyn();
        let decoder_input = CowArray::from(Array2::zeros((bs, n_mel_channels))).into_dyn();
        // This is only really needed for batched inputs
        let mask = CowArray::from(Array2::from_elem((1, seq_len), false)).into_dyn();

        Self {
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            decoder_input,
            mask,
        }
    }
}

impl Tacotron2 {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let environment = Environment::builder()
            .with_name("xd_tts")
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .build()?
            .into_arc();

        let encoder = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_model_from_file(path.as_ref().join("encoder.onnx"))
            .context("converting encoder to runnable model")?;

        let decoder = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_model_from_file(path.as_ref().join("decoder_iter.onnx"))
            .context("converting decoder_iter to runnable model")?;

        let post_net = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_model_from_file(path.as_ref().join("postnet.onnx"))
            .context("converting postnet to runnable model")?;

        Ok(Self {
            encoder,
            decoder,
            post_net,
            phoneme_ids: generate_id_list(),
        })
    }

    fn run_decoder<'a>(
        &self,
        memory: &'a CowArray<'a, f32, IxDyn>,
        processed_memory: &'a CowArray<'a, f32, IxDyn>,
        state: &'a mut DecoderState<'a>,
    ) -> anyhow::Result<()> {
        let alloc = self.decoder.allocator();

        let gate_threshold = 0.6;
        let max_decoder_steps = 1000;

        for i in 1..max_decoder_steps {
            let inputs = vec![
                Value::from_array(alloc, &state.decoder_input)?,
                Value::from_array(alloc, &state.attention_hidden)?,
                Value::from_array(alloc, &state.decoder_hidden)?,
                Value::from_array(alloc, &state.decoder_cell)?,
                Value::from_array(alloc, &state.attention_weights)?,
                Value::from_array(alloc, &state.attention_weights_cum)?,
                Value::from_array(alloc, &state.attention_context)?,
                Value::from_array(alloc, memory)?,
                Value::from_array(alloc, processed_memory)?,
                Value::from_array(alloc, &state.mask)?,
            ];

            // init decoder inputs
            let infer = self.decoder.run(inputs)?;
        }

        todo!()
    }

    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        let mut phonemes = units
            .iter()
            .map(|x| best_match_for_unit(x, &self.phoneme_ids))
            .collect::<Vec<_>>();

        // So it's not documented or shown in the inference functions but if your tensor is a lower
        // sequence length than the LSTM node in the encoder it will fail. This length is 50 (seen
        // via netron) so here I just pad it to 50 if it's below.
        if phonemes.len() < 50 {
            phonemes.resize(50, 0);
        }

        // Run encoder
        info!("{:?}", phonemes.len());
        let plen = CowArray::from(arr1(&[phonemes.len() as i64])).into_dyn();
        let phonemes =
            Array2::from_shape_vec((1, phonemes.len()), phonemes).context("invalid dimensions")?;
        let phonemes = CowArray::from(phonemes).into_dyn();

        let inputs = vec![
            Value::from_array(self.encoder.allocator(), &phonemes)?,
            Value::from_array(self.encoder.allocator(), &plen)?,
        ];

        let encoder_outputs = self.encoder.run(inputs)?;
        assert_eq!(encoder_outputs.len(), 3);
        info!("{:?}", encoder_outputs);

        // The outputs in order are: memory, processed_memory, lens. Despite the name
        // OrtOwnedTensor
        let memory: OrtOwnedTensor<f32, _> = encoder_outputs[0].try_extract()?;
        let processed_memory: OrtOwnedTensor<f32, _> = encoder_outputs[1].try_extract()?;
        let lens: OrtOwnedTensor<i64, _> = encoder_outputs[2].try_extract()?;

        let mut decoder_state =
            DecoderState::new(&memory.view(), &processed_memory.view(), &lens.view());

        let memory = CowArray::from(memory.view().to_owned());
        let processed_memory = CowArray::from(processed_memory.view().to_owned());

        self.run_decoder(&memory, &processed_memory, &mut decoder_state)?;

        todo!()
    }
}
