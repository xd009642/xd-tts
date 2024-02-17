import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.io.wavfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='plot_spectrograms',
                    description='Plots two spectrograms for comparison')

    parser.add_argument("python", type=str, help="path to npy file to plot on the left side")  
    parser.add_argument("rust", type=str, help="path to npy file to plot on the left side")  
    parser.add_argument("--out", "-o", type=str, help="path to save figure to", default="melgen_py_vs_rust.svg")  
    args = parser.parse_args()


    spec = np.load(args.python)
    spec = spec.squeeze()

    print(spec.shape)

    fig, ax = plt.subplots(nrows=2, sharex=False, sharey=False)

    librosa.display.specshow(spec, ax=ax[0])

    spec = np.load(args.rust)
    spec = spec.squeeze()

    print(spec.shape)

    librosa.display.specshow(spec, ax=ax[1])

    ax[0].set_title("Python Output")
    ax[1].set_title("Rust ONNX Output")

    plt.savefig(args.out)
    plt.show()

