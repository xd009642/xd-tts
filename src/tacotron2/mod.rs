use crate::phonemes::*;
use anyhow::Context;
use griffin_lim::mel::create_mel_filter_bank;
use griffin_lim::GriffinLim;
use ndarray::Array2;
use ndarray::{concatenate, prelude::*};
use ort::{inputs, CPUExecutionProvider, GraphOptimizationLevel, Session, Tensor};
use std::path::Path;
use std::str::FromStr;
use tracing::debug;

// Mel parameters:
// fmin 0
// fmax 7000
// win-length 1024
// hop-length 256
// filter length 1024 (number of FFTs)
// sample rate 22050
// number of mels 80
//
// How many FFT bins?
//
// So the paper says fmin-fmax are 125Hz to 7.6kHz

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

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let x = -x;
        1.0 / (1.0 + x.exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

// Downloaded from `https://developer.nvidia.com/joc-tacotron2-fp32-pyt-20190306` and used
// `export_tacotron2_onnx.py` in https://github.com/NVIDIA/DeepLearningExamples
pub struct Tacotron2 {
    encoder: Session,
    decoder: Session,
    postnet: Session,
    phoneme_ids: Vec<Unit>,
}

struct DecoderState {
    decoder_input: Array2<f32>,
    attention_hidden: Array2<f32>,
    attention_cell: Array2<f32>,
    decoder_hidden: Array2<f32>,
    decoder_cell: Array2<f32>,
    attention_weights: Array2<f32>,
    attention_weights_cum: Array2<f32>,
    attention_context: Array2<f32>,
    //    memory: CowArray<f32, Ix3>,
    //    processed_memory: CowArray<f32, Ix3>,
    mask: Array2<bool>,
}

impl DecoderState {
    fn new(memory: &ArrayViewD<f32>, unpadded_len: usize) -> Self {
        let bs = memory.shape()[0];
        let seq_len = memory.shape()[1];
        let attention_rnn_dim = 1024;
        let decoder_rnn_dim = 1024;
        let encoder_embedding_dim = 512;
        let n_mel_channels = 80;

        let attention_hidden = Array2::zeros((bs, attention_rnn_dim));
        let attention_cell = Array2::zeros((bs, attention_rnn_dim));
        let decoder_hidden = Array2::zeros((bs, decoder_rnn_dim));
        let decoder_cell = Array2::zeros((bs, decoder_rnn_dim));
        let attention_weights = Array2::zeros((bs, seq_len));
        let attention_weights_cum = Array2::zeros((bs, seq_len));
        let attention_context = Array2::zeros((bs, encoder_embedding_dim));
        let decoder_input = Array2::zeros((bs, n_mel_channels));
        // This is only really needed for batched inputs
        let mut mask = Array2::from_elem((1, seq_len), false);
        mask.slice_mut(s![.., unpadded_len..]).fill(true);

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
        // ort calls into a C++ library which has it's own global initialisation that needs to be
        // ran. Fortunately, this can be called multiple times so we don't have to fiddle around to
        // make it safer.
        ort::init()
            .with_name("xd_tts")
            .with_execution_providers(&[CPUExecutionProvider::default().build()])
            .commit()?;

        // Load all the networks. Context is added to the error so we can tell easily which network
        // messes things up

        let encoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_model_from_file(path.as_ref().join("encoder.onnx"))
            .context("converting encoder to runnable model")?;

        let decoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_model_from_file(path.as_ref().join("decoder_iter.onnx"))
            .context("converting decoder_iter to runnable model")?;

        let postnet = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_model_from_file(path.as_ref().join("postnet.onnx"))
            .context("converting postnet to runnable model")?;

        Ok(Self {
            encoder,
            decoder,
            postnet,
            phoneme_ids: generate_id_list(),
        })
    }

    fn run_decoder(
        &self,
        memory: &Array<f32, IxDyn>,
        processed_memory: &Array<f32, IxDyn>,
        state: &mut DecoderState,
    ) -> anyhow::Result<Array2<f32>> {
        let gate_threshold = 0.6;
        let max_decoder_steps = 1000;

        let mut inputs = inputs![
            "decoder_input" => state.decoder_input.view(),
            "attention_hidden" => state.attention_hidden.view(),
            "attention_cell" => state.attention_cell.view(),
            "decoder_hidden" => state.decoder_hidden.view(),
            "decoder_cell" => state.decoder_cell.view(),
            "attention_weights" => state.attention_weights.view(),
            "attention_weights_cum" => state.attention_weights_cum.view(),
            "attention_context" => state.attention_context.view(),
            "memory" => memory.view(),
            "processed_memory" => processed_memory.view(),
            "mask" => state.mask.view()
        ]?;
        // Concat the spectrogram etc

        let mut mel_spec = Array2::zeros((0, 0));

        // Because we always break out of this we could use `loop`.
        for i in 0..max_decoder_steps {
            // init decoder inputs
            let mut infer = self.decoder.run(inputs)?;

            let gate_prediction = &infer["gate_prediction"].extract_tensor::<f32>()?;
            let mel_output = &infer["decoder_output"].extract_tensor::<f32>()?;
            let mel_output = mel_output.view().clone().into_dimensionality()?;

            debug!("Gate: {}", gate_prediction.view()[[0, 0]]);

            if i == 0 {
                mel_spec = mel_output.to_owned();
            } else {
                mel_spec = concatenate(Axis(0), &[mel_spec.view(), mel_output.view()])?;
            }

            if sigmoid(gate_prediction.view()[[0, 0]]) > gate_threshold
                || i + 1 == max_decoder_steps
            {
                debug!("Stopping after {} steps", i);
                break;
            }
            // Prepare the inputs for the next run. We could put this in a condition, but as it's
            // moved on inference it's hard to do this and keep the borrow checker happy. So I
            // moved the condition up to above with the break.
            inputs = inputs![
                "memory" => memory.view(),
                "processed_memory" => processed_memory.view(),
                "mask" => state.mask.view(),
            ]?;
            inputs.insert("decoder_input", infer.remove("decoder_output").unwrap());
            inputs.insert(
                "attention_hidden",
                infer.remove("out_attention_hidden").unwrap(),
            );
            inputs.insert(
                "attention_cell",
                infer.remove("out_attention_cell").unwrap(),
            );
            inputs.insert(
                "decoder_hidden",
                infer.remove("out_decoder_hidden").unwrap(),
            );
            inputs.insert("decoder_cell", infer.remove("out_decoder_cell").unwrap());
            inputs.insert(
                "attention_weights",
                infer.remove("out_attention_weights").unwrap(),
            );
            inputs.insert(
                "attention_weights_cum",
                infer.remove("out_attention_weights_cum").unwrap(),
            );
            inputs.insert(
                "attention_context",
                infer.remove("out_attention_context").unwrap(),
            );
        }

        // We have to transpose it and add in a batch dimension for it to be the right shape.
        let mel_spec = mel_spec.t().insert_axis(Axis(0));

        let post = self.postnet.run(inputs![mel_spec.view()]?)?;

        let post = post["mel_outputs_postnet"]
            .extract_tensor::<f32>()?
            .view()
            .clone()
            .remove_axis(Axis(0))
            .into_dimensionality()?
            .into_owned();

        Ok(post)
    }

    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        let mut phonemes = units
            .iter()
            .map(|x| best_match_for_unit(x, &self.phoneme_ids))
            .collect::<Vec<_>>();

        debug!("{:?}", phonemes);

        // So it's not documented or shown in the inference functions but if your tensor is a lower
        // sequence length than the LSTM node in the encoder it will fail. This length is 50 (seen
        // via netron) so here I just pad it to 50 if it's below. This is likely due to torch JIT
        // replacing some dynamic values with constant ones!
        if phonemes.len() < 100 {
            phonemes.resize(100, 0);
        }

        // Run encoder
        debug!("{:?}", phonemes.len());
        let plen = arr1(&[phonemes.len() as i64]);
        let phonemes =
            Array2::from_shape_vec((1, phonemes.len()), phonemes).context("invalid dimensions")?;

        let encoder_outputs = self.encoder.run(inputs![phonemes, plen]?)?;
        assert_eq!(encoder_outputs.len(), 3);

        // The outputs in order are: memory, processed_memory, lens. Despite the name
        // OrtOwnedTensor
        let memory: Tensor<f32> = encoder_outputs[0].extract_tensor()?;
        let processed_memory: Tensor<f32> = encoder_outputs[1].extract_tensor()?;

        let mut decoder_state = DecoderState::new(&memory.view(), units.len());

        let memory = memory.view().to_owned();
        let processed_memory = processed_memory.view().to_owned();

        self.run_decoder(&memory, &processed_memory, &mut decoder_state)
    }
}

/// Creates a griffin-lim vocoder for the tacotron2 model
pub fn create_griffin_lim() -> anyhow::Result<GriffinLim> {
    // So these parameters we get from the config.json in the tacotron2 repo that lets us know
    // what parameters they're using for their vocoder. They're also available here:
    // https://catalog.ngc.nvidia.com/orgs/nvidia/resources/tacotron_2_and_waveglow_for_pytorch/advanced
    //
    // For momentum the default parameter from the librosa implementation is used, this was not
    // tweaked as it delivered reasonable results. For the power this was tuned by ear. The
    // spectrograms that come out of mel-gen models are sometimes a bit quiet so tuning based on
    // the model is vital. For tacotron2 it seems a value around 1.2-1.7 is the best.
    //
    // For iterations there wasn't any perceivable increase in quality after 10 iterations, but as
    // it's fast I kept it at 20 just in case there's some trickier/noisier samples.
    let mel_basis = create_mel_filter_bank(22050.0, 1024, 80, 0.0, Some(8000.0));
    // So the hop length is 256, this means the overlap is the window_size - hop_length. Getting
    // this value wrong will result in noisier time stretched versions of the audio.
    let vocoder = GriffinLim::new(mel_basis, 1024 - 256, 1.7, 20, 0.99)?;
    Ok(vocoder)
}
