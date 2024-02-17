//! Tacotron2 is a encoder-decoder sequence to sequence RNN model which predicts a sequence of mel
//! spectrogram frames from a sequence of tokens. These can be characters and the pre-trained
//! models provided are trained on characters, but they can also be phonemes. And while phonemes
//! provide more control over pronunciation, they require another stage between text processing
//! and spectrogram generation (g2p, dict lookup etc).
//!
//! Also, tacotron2 has been shown to do some interesting things with emphasis by getting users in
//! data collection to stress the capitalised word. So the character input is also case sensitive.
//!
//! This does add some complexity though, the nvidia pretrained model is trained only on lowercase
//! characters and punctuation. If you provide upper case characters or phonemes to the model it
//! will generate gibberish. You can work out the lowercase character part from the code, but not
//! the phoneme part (it is mentioned in a github issue or forum somewhere).
//!
//! If you want to learn more about tacotron2 here are some resources
//!
//! * [Papers with code](https://paperswithcode.com/method/tacotron-2)
//! * [Publication website](https://google.github.io/tacotron/publications/tacotron2/)
//! * [nvidia
//! catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/tacotron_2_and_waveglow_for_pytorch)
//!
//! This module primarily deals with
//!
//! 1. Tacotron2 inference, converting units to IDs, running networks, getting output
//! 2. Making the output usable
//!
//! The latter part is accomplished via small utility functions, things like getting a vocoder for
//! the model with the correct parameters to generate audio. If these parameters are wrong then the
//! audio will sound pitch shifted, like random noise or demonic nasal demons.
//!
//! Some things have been done quickly or within the code to try and keep things constrained to the
//! same file. In general I'd recommend every neural network come with some sort of config file
//! (json etc) which details things like these parameters, the IDs for the input/output and
//! relevant tensor names. This allows faster swapping of models if researchers are playing around
//! with different training or model setups. For keeping to one known architecture the returns can
//! diminish especially if you're not training your own model.
//!
//! # What's is an Encoder-Decoder Model?
//!
//! An encoder-decoder model is a sequence-to-sequence model, this means it maps from one sequence
//! to another sequence of potentially varying length. The encoder-decoder model is a specific
//! neural network architecture which allows this to happen.
//!
//! Traditional approaches worked well if the alignment between the input and output sequences
//! was known ahead of time and the order was the same for input and output sequence. The ordering
//! is the same for the sequences with TTS, but the durations aren't known. For each phoneme we
//! input the length of the audio for that phoneme may differ.
//!
//! The encoder part of the model will output some internal state which will typically be a vector
//! representation of the input, and other parameters that may be used to determine the output
//! sequence length. We can see in other TTS models the phoneme durations are output with a vector
//! representation, then the phoneme durations are used to determine how long the decoder runs.
//!
//! The decoder takes the state, then runs it through until completion. Often encoder-decoder
//! architectures maintain some form of count or End-of-Sequence token to determine when to stop.
//! Tacotron2 is a pre-transformers version of encoder-decoder, for each decoding step the encoder
//! output gets weighted by an attention module to ensure it progresses through the sequence and
//! doesn't get stuck. You can read one of the papers about this mechanism
//! [here](https://arxiv.org/pdf/1506.07503.pdf).
//!
//! If you're interested in more in depth explanations look for sequence-to-sequence learning and
//! LSTMs. You can see these architectures appear in other audio related tasks such as
//! transcription and also in machine translation. These areas as well as TTS will refer to a lot
//! of related foundational knowledge.
use crate::phonemes::*;
use anyhow::Context;
use griffin_lim::mel::create_mel_filter_bank;
use griffin_lim::GriffinLim;
use ndarray::Array2;
use ndarray::{concatenate, prelude::*};
use ort::{inputs, CPUExecutionProvider, GraphOptimizationLevel, Session, Tensor};
use std::path::Path;
use std::str::FromStr;
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

/// Function to generate the ordered unit ID list for tacotron2. Any character/punctuation/phoneme
/// can be searched in this list and it's index will correspond to the model input.
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

/// Sigmoid function, would have been done by the network but the ONNX split meant it was no
/// longer part of the graph.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let x = -x;
        1.0 / (1.0 + x.exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

/// Handle to the tacotron2 ONNX graphs.
///
/// These were initially downloaded from `https://developer.nvidia.com/joc-tacotron2-fp32-pyt-20190306` and used
/// `export_tacotron2_onnx.py` in https://github.com/NVIDIA/DeepLearningExamples
pub struct Tacotron2 {
    /// Encoder part of the transformer
    encoder: Session,
    /// Decoder update part
    decoder: Session,
    /// A post network to adjust the outputs
    postnet: Session,
    /// IDs of the input tokens
    phoneme_ids: Vec<Unit>,
}

/// We don't want to trigger clippy warnings about too many parameters so the decoder state ran
/// through each update step is kept in a struct. This also makes the part of the code passing it
/// around easier to read than a mess of parameters where all types are the same!
///
/// Details for what this state does can be found within the paper/code (in varying levels of
/// details). But I'll attempt to roughly summarise at a top level.
///
/// The mask is the simplest field. It's false for every element of the input sequence and true
/// once the end is hit. This is because when you batch up multiple inputs at once you want the
/// network to stop at the end of the sequence and not overrun it for a sample because there's a
/// longer sample in the batch (tensors are dense not sparse).
///
/// The attention weight fields: _"encourages the model to move forward consistently through the
/// input"_. This quote is taken from [the paper](https://arxiv.org/pdf/1712.05884.pdf) and
/// suggests there's an attention mechanism where the network can move forward or backward through
/// the sequence and therefore needs guiding in the right direction.
///
/// From the encoder output we get the size of the spectrogram to generate. This is used to create
/// the decoder tensors which are iteratively updated by the decoder network, this is like a
/// scratch pad to store the network output and any working data it needs to keep generating that
/// output. The decoder input and output are sized based on output audio length and number of mel
/// spectrograms, the decoder hidden and cell fields are sized based on weight dimensions and
/// sequence length so must be storing the state of the neurons to feedback into the model.
///
/// A key node in the Tacotron2 model is the
/// [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) a "long short-term memory"
/// cell designed to work on sequences of data.
struct DecoderState {
    /// Input to the decoder LSTM
    decoder_input: Array2<f32>,
    /// Hidden state of the attention LSTM node
    attention_hidden: Array2<f32>,
    /// Cell state of the attention LSTM node
    attention_cell: Array2<f32>,
    /// Hidden state of the decoder LSTM node
    decoder_hidden: Array2<f32>,
    /// Cell state of the decoder LSTM node
    decoder_cell: Array2<f32>,
    /// Atteention weights used to guide the model towards the correct part of the sequence.
    attention_weights: Array2<f32>,
    /// Cumulative weights of the attention mechanism
    attention_weights_cum: Array2<f32>,
    /// Output of the attention part of the decoder, this is fed into the decoding part to generate
    /// the decoder output. And kept here so we can feed it into the attention on the next step.
    attention_context: Array2<f32>,
    /// Used to denote length of sequence
    mask: Array2<bool>,
}

impl DecoderState {
    /// Creates a new decoder state given the output of the encoder network and the length of the
    /// sequence before padding.
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
    /// Load a tacotron2 model from a folder. This folder should contain 3 files:
    ///
    /// 1. encoder.onnx
    /// 2. decoder_iter.onnx
    /// 3. postnet.onnx
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

    /// Run the decoder stage of the network. This function would be fairly small if not for the
    /// amount of state that needs to be extracted from the model and fed into it, however it is
    /// relatively low complexity.
    fn run_decoder(
        &self,
        memory: &Array<f32, IxDyn>,
        processed_memory: &Array<f32, IxDyn>,
        state: &mut DecoderState,
    ) -> anyhow::Result<Array2<f32>> {
        // Constants taken from the python implementation
        let gate_threshold = 0.6;
        let max_decoder_steps = 1000;

        // An example of why setting inputs based on names is much more readable to someone
        // approaching ML code.
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
                mel_spec = concatenate(Axis(0), &[mel_spec.view(), mel_output.view()])
                    .context("Joining decoder iter output")?;
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

    /// Given a chunk of phonemes run inference
    fn infer_chunk(&self, mut phonemes: Vec<i64>) -> anyhow::Result<Array2<f32>> {
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

        let mut decoder_state = DecoderState::new(&memory.view(), units_len);

        let memory = memory.view().to_owned();
        let processed_memory = processed_memory.view().to_owned();

        self.run_decoder(&memory, &processed_memory, &mut decoder_state)
    }

    /// Runs inference on the units returning a mel-spectrogram. This will split the inference into
    /// smaller chunks that fit into the models fixed size input window and run as many inferences
    /// as necessary.
    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        let mut splits = find_splits(units, 100);

        // There's no UNK input to tacotron2, so we're just going to silently throw away failing
        // units (do not do this in a real system)
        let mut phonemes = units
            .iter()
            .filter_map(|x| best_match_for_unit(x, &self.phoneme_ids))
            .collect::<Vec<_>>();
        info!("Phonemes: {:?}", phonemes);

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
