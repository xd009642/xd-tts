#import "@preview/polylux:0.3.1": *
#import "theme.typ": *;

#show: talk-theme

#set page(paper: "presentation-16-9")
#set text(size: 25pt)

#let include_speedy_speech = false

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
  == What is a Mel Spectrogram?

  - The mel scale is a pitch scale so that all tones sound equidistant to human ears
  - For a window of time you can think of this as a histogram of frequency information
  - The smaller a feature space the easier to fit a network to at the cost of accuracy
  - Generating raw audio would require a lot more data, quantising in terms of pitch and time reduces the training cost
  - We can also limit ourselves to pitches in audible range and avoid fitting to high or low frequency noise in the data
]

#slide[
  == A Note on Neural Networks

  - So here we're going to avoid using Tensorflow or Torch
  - Why? Because it's more interesting (I hope)
  - It also lets us look at more of the Rust Ecosystem including runtimes which can run on more devices
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
#if include_speedy_speech [

    #slide[
      == SpeedySpeech

      - Initially tried speedyspeech as a "lightweight CPU suitable mel-gen"
      - Old version of torch so needed to patch a lot to export to ONNX
      - And torch doesn't offer an easy upgrade path for models - researcher advice is "don't do this"
      - Failed to get it into every ONNX supporting framework
      - Abandoned in favour of Tacotron2 a more popular slightly newer model
    ]

    #slide[
      == So what went wrong?

      - SpeedySpeech uses variable length tensors for internal attention. Very poor support for this
      - Also loops are poorly supported in the ecosystem
      - Tract got closest but those two requirements meant changes to the network were needed
      - Dfdx no ONNX support yet
      - Burn tries to turn onnx into rust code in a build.rs panicked with unhelpful error
      - Candle support is very new (git dependency new), missing a lot of features
      - In the end I would have had to rework the architecture and retrain the network to use it
    ]

]

#slide[
  == Tacotron2

  - Sequence-to-sequence model, published 2018. 
  - No longer state of the art - but still very good
  - ONNX export splits the network into 3 subnetworks
  - This is because of generally poor ONNX support in the ML ecosystem
  - Waveglow ONNX export doesn't even succeed, it panics instead during export!
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
 - Both have us using ndarray but tract forces wrapping it into their Tensor and TValue types
 - Tract feels more idiomatic Rust and is easier to use, but Tensor vs TValue adds friction.
]
