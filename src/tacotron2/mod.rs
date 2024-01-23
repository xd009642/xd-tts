use crate::phonemes::*;
use anyhow::Context;
use griffin_lim::mel::create_mel_filter_bank;
use griffin_lim::GriffinLim;
use ndarray::Array2;
use ndarray::{concatenate, prelude::*};
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::InferenceOp;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::rc::Rc;
use tracing::{debug, info};

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

type Model = SimplePlan<InferenceFact, Box<dyn InferenceOp>, Graph<InferenceFact, Box<dyn InferenceOp>>>;

// Downloaded from `https://developer.nvidia.com/joc-tacotron2-fp32-pyt-20190306` and used
// `export_tacotron2_onnx.py` in https://github.com/NVIDIA/DeepLearningExamples
pub struct Tacotron2 {
    encoder: Model,
    decoder: Model,
    postnet: Model,
    phoneme_ids: Vec<Unit>,
}

struct DecoderState {
    decoder_input: TValue,
    attention_hidden: TValue,
    attention_cell: TValue,
    decoder_hidden: TValue,
    decoder_cell: TValue,
    attention_weights: TValue,
    attention_weights_cum: TValue,
    attention_context: TValue,
    mask: TValue,
}

impl DecoderState {
    fn new(memory: &Tensor, unpadded_len: usize) -> anyhow::Result<Self> {
        let bs = memory.shape()[0];
        let seq_len = memory.shape()[1];
        let attention_rnn_dim = 1024;
        let decoder_rnn_dim = 1024;
        let encoder_embedding_dim = 512;
        let n_mel_channels = 80;

        let attention_hidden = TValue::from_const(Arc::new(Tensor::zero::<f32>(&[bs, attention_rnn_dim])?));
        let attention_cell = attention_hidden.clone();
        let decoder_hidden = TValue::from_const(Arc::new(Tensor::zero::<f32>(&[bs, decoder_rnn_dim])?));
        let decoder_cell = decoder_hidden.clone();
        let attention_weights = TValue::from_const(Arc::new(Tensor::zero::<f32>(&[bs, seq_len])?));
        let attention_weights_cum = attention_weights.clone();
        let attention_context = TValue::from_const(Arc::new(Tensor::zero::<f32>(&[bs, encoder_embedding_dim])?));
        let decoder_input = TValue::from_const(Arc::new(Tensor::zero::<f32>(&[bs, n_mel_channels])?));
        // This is only really needed for batched inputs
        let mut mask = Array2::from_elem((1, seq_len), false);
        mask.slice_mut(s![.., unpadded_len..]).fill(true);

        let mask = TValue::from_const(Arc::new(mask.into()));

        Ok(Self {
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            decoder_input,
            mask,
        })
    }
}

impl Tacotron2 {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {

        let encoder = tract_onnx::onnx()
            .model_for_path(path.as_ref().join("encoder.onnx"))
            .context("loading encoder onnx")?
            .into_runnable()
            .context("creating runnable encoder")?;
        
        let decoder = tract_onnx::onnx()
            .model_for_path(path.as_ref().join("decoder_iter.onnx"))
            .context("loading decoder onnx")?
            .into_runnable()
            .context("creating runnable decoder")?;
        
        let postnet = tract_onnx::onnx()
            .model_for_path(path.as_ref().join("postnet.onnx"))
            .context("loading postnet onnx")?
            .into_runnable()
            .context("creating runnable postnet")?;

        Ok(Self {
            encoder,
            decoder,
            postnet,
            phoneme_ids: generate_id_list(),
        })
    }

    fn run_decoder(
        &self,
        memory: TValue,
        processed_memory: TValue,
        state: DecoderState,
    ) -> anyhow::Result<Array2<f32>> {
        let gate_threshold = 0.6;
        let max_decoder_steps = 1000;

        let mut inputs = tvec![
            state.decoder_input,
            state.attention_hidden,
            state.attention_cell,
            state.decoder_hidden,
            state.decoder_cell,
            state.attention_weights,
            state.attention_weights_cum,
            state.attention_context,
            memory.clone(),
            processed_memory.clone(),
            state.mask.clone(),
        ];
        // Concat the spectrogram etc

        let mut mel_spec = Array2::zeros((0, 0));

        // Because we always break out of this we could use `loop`.
        for i in 0..max_decoder_steps {
            debug!("Decoder iter: {}", i);
            // init decoder inputs
            let mut infer = self.decoder.run(inputs)?;

            let gate_prediction = infer.remove(1);
            let gate_prediction = *gate_prediction.to_scalar::<f32>()?;
            let mel_output = &infer[0];
            let mel_output = mel_output.to_array_view::<f32>()?
                .clone()
                .into_dimensionality()?;

            debug!("Gate: {}", gate_prediction);

            if i == 0 {
                mel_spec = mel_output.to_owned();
            } else {
                mel_spec = concatenate(Axis(0), &[mel_spec.view(), mel_output.view()])
                    .context("Joining decoder iter output")?;
            }

            if sigmoid(gate_prediction) > gate_threshold
                || i + 1 == max_decoder_steps
            {
                debug!("Stopping after {} steps", i);
                break;
            }
            // Prepare the inputs for the next run. We could put this in a condition, but as it's
            // moved on inference it's hard to do this and keep the borrow checker happy. So I
            // moved the condition up to above with the break.
            inputs = infer;
            inputs.push(memory.clone());
            inputs.push(processed_memory.clone());
            inputs.push(state.mask.clone());
        }

        // We have to transpose it and add in a batch dimension for it to be the right shape.
        let mel_spec = mel_spec.t().insert_axis(Axis(0)).into_owned();

        let mel_spec = TValue::Var(Rc::new(mel_spec.into()));
        let post = self.postnet.run(tvec![mel_spec])?;

        let post = post[0]
            .to_array_view::<f32>()?
            .clone()
            .remove_axis(Axis(0))
            .into_dimensionality()?
            .into_owned();

        Ok(post)
    }

    fn infer_chunk(&self, mut phonemes: Vec<i64>) -> anyhow::Result<Array2<f32>> {
        debug!("Running {} phonemes pre padding", phonemes.len());
        let units_len = phonemes.len();
        assert!(units_len <= 100);

        // So it's not documented or shown in the inference functions but if your tensor is a lower
        // sequence length than the LSTM node in the encoder it will fail. This length is 50 (seen
        // via netron) so here I just pad it to 50 if it's below. This is likely due to torch JIT
        // replacing some dynamic values with constant ones!
        if phonemes.len() < 100 {
            phonemes.resize(100, 0);
        }

        // Run encoder
        let plen = Tensor::from_shape(&[1], &[phonemes.len() as i64])?;
        let plen = TValue::from_const(Arc::new(plen));
        let phonemes =
            Array2::from_shape_vec((1, phonemes.len()), phonemes).context("invalid dimensions")?;
        let phonemes = TValue::from_const(Arc::new(phonemes.into()));

        debug!("Starting encoder inference");
        let mut encoder_outputs = self.encoder.run(tvec![phonemes, plen])?;
        debug!("Finished encoder inference");
        assert_eq!(encoder_outputs.len(), 3);

        // The outputs in order are: memory, processed_memory, lens. Despite the name
        // OrtOwnedTensor
        let memory = encoder_outputs.remove(0);
        let processed_memory = encoder_outputs.remove(0);

        let decoder_state = DecoderState::new(&memory, units_len)?;

        self.run_decoder(memory, processed_memory, decoder_state)
    }

    /// Runs inference on the units returning a mel-spectrogram
    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        let mut splits = find_splits(units, 100);

        let mut phonemes = units
            .iter()
            .map(|x| best_match_for_unit(x, &self.phoneme_ids))
            .collect::<Vec<_>>();

        let mut mel_spec = Array2::zeros((0, 0));

        // Make sure we have at least one because of the lazy split implementation.
        if !splits.contains(&units.len()) {
            splits.push(units.len());
        }
        info!("Splits: {:?}", splits);

        let mut offset = 0;
        // So interestingly if we exceed the input length we end up getting silence back. Instead
        // of spending too much time debugging this I'm going to ensure we stick to the fixed
        // length as our ONNX has a fixed input size and we're not going to be giving dynamic sized
        // inputs to a fixed size tensor.
        for split in splits.iter() {
            let remaining = phonemes.split_off(*split - offset);
            offset += phonemes.len();
            let array = self.infer_chunk(phonemes)?;

            if mel_spec.is_empty() {
                mel_spec = array;
            } else {
                mel_spec = concatenate(Axis(1), &[mel_spec.view(), array.view()])
                    .context("Joining inference chunk output")?;
            }
            phonemes = remaining;
        }

        Ok(mel_spec)
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
    let vocoder = GriffinLim::new(mel_basis, 1024 - 256, 1.7, 30, 0.99)?;
    Ok(vocoder)
}
