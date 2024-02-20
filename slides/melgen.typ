#import "@preview/polylux:0.3.1": *
#import "theme.typ": *;

#show: talk-theme

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#focus-slide[
  == Spectrogram Generation
]

#slide[
  == Why Generate a Spectrogram?

  - Generally in AI the more we can constrain a problem the easier it is to make a network converge
  - Higher dimensionality data input or output requires more data to train
  - So instead of generating audio we generate a representation with much less data
]

#slide[
  == What Is a Mel Spectrogram?

  - The mel scale is a pitch scale so that all tones sound equidistant to human ears
  - For a window of time you can think of this as a histogram of frequency information
  - The smaller a feature space the easier to fit a network to at the cost of accuracy
  - Generating raw audio would require a lot more data, quantising in terms of pitch and time reduces the training cost
]

#slide[
  == A Note on Neural Networks

  - So here we're going to avoid using Tensorflow or Torch
  - Why? Because it's more interesting (I hope)
  - It also lets us look at more of the Rust Ecosystem including runtimes which can run on more devices
]

#slide[
  == Tacotron2

  - Sequence-to-sequence model, published 2018. 
  - No longer state of the art - but still very good
]

#slide[
  == ONNX

  - Open Neural Network Exchange
  - A format to make it easier to run Neural Networks outside of the frameworks you trained them in
  - Adoption feels poor and ecosystem feels lacking
  - But when it works it's great
  - Best native rust support is in tract
  - ort is bindings to the official runtime (C++) and is fully featured 
]

#slide[
    == Useful ONNX Tools

    - https://netron.app/ visualise the ONNX
    - https://github.com/onnx/optimizer optimise the graphs for inference speed
    - https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html graph surgeon, introspect and manipulate ONNX graphs
]

#slide[
    == ONNX and Tacotron2
  
  - ONNX export splits the network into 3 subnetworks
  - This is because of generally poor ONNX support in the ML ecosystem
  - The ONNX export for default Tacotron2 vocoder doesn't even succeed, it panics instead during export!
]

#slide[
  == Picking an ONNX Runtime

  - Need to pick an ONNX Runtime
  - Must support the necessary operations and features
  - Performance should also be acceptable 
  - I'll accept a bit slower for pure Rust
  - Ended up with a decision between Tract and ort
]

#slide[
  == Tract

  - Tract pure Rust and best spec support in the Rust ML ecosystem
  - Missing loop blocks and dynamically sized inputs
  - Optimising interpretter approach to ONNX
  - Also inference speed isn't competitive with non-Rust competitors
  - Real Time Factor of ~300 on "Hello world from Rust"
]

#slide[
  == Tract

#text(size: 18pt)[
```rust
type Model = SimplePlan<InferenceFact, Box<dyn InferenceOp>, Graph<InferenceFact, Box<dyn InferenceOp>>>;

pub struct Tacotron2 {
   encoder: Model,
}

let encoder = tract_onnx::onnx()
    .model_for_path(path.as_ref().join("encoder.onnx"))?
    .into_runnable()?;

let phonemes = TValue::from_const(Arc::new(phonemes.into()));
let plen = Tensor::from_shape(&[1], &[phonemes.len() as i64])?;
let encoder_output = self.encoder.run(tvec![phonemes, plen])?;
```
]
]

#slide[
  == ORT

  - ONNX RunTime. Bindings to Microsoft's C++ ONNX runtime
  - Best spec support in the wider ML ecosystem
  - Decent performance
  - Can perform graph optimisations
  - Real Time Factor of ~0.5 on "Hello world from Rust"
]

#slide[
  == ORT

#text(size: 18pt)[
```rust
pub struct Tacotron2 {
    encoder: Session,
}

ort::init()
    .with_execution_providers(&[CPUExecutionProvider::default().build()])
    .commit()?;
let encoder = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level1)?
    .with_model_from_file(path.as_ref().join("encoder.onnx"))?;

let plen = arr1(&[phonemes.len() as i64]);
// also allows inputs!["phonemes"=> phonemes.view(), "plen" => plen.view()]
let encoder_outputs = self.encoder.run(inputs![phonemes, plen]?)?;
```
]
]

#slide[
 == Thoughts

 - ORT API is lower level, harder to use
 - But being able to specify inputs by name is really nice!
 - Both have us using ndarray but tract forces wrapping it into their `Tensor` and `TValue` types
 - Tract feels more idiomatic Rust and is easier to use, but `Tensor` vs `TValue` adds friction.
]

#slide[
    == Why are Named Tensor Inputs/Outputs Useful?

    #set text(size: 15pt)
    ```rust
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
```
]

#slide[
  == Changes

  - So after export inference is different in our Rust code
  - We need to manually run the decoder iter
  - We also need to maintain the state each loop
  - The dynamic input dimension is now fixed because of JIT tracing
  - This means the mask input has to be changed from the Python implementation
  - The outputs between Python and Rust don't look the same
]

#focus-slide[
  == But They Look Close!
]

#slide[
  #align(center)[#image("images/melgen_py_vs_rust.svg")]
]

#slide[
    == Don't Trust Researcher Documentation

    - Tacotron2's text processing says it can take uppercase/lowercase characters or ARPABET
    - But the pretrained models weren't trained with any ARPABET or uppercase characters
    - You'll get weird output!
]
