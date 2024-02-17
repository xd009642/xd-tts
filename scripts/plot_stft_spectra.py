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

    parser.add_argument("--input", "-i", type=str, help="path to npy file to plot on the left side")  
    parser.add_argument("--output", "-o", type=str, help="path to save figure to", default="stft.svg")  
    
    args = parser.parse_args()

    y, sr = librosa.load(args.input)
    D = librosa.stft(y)
    S, phi = np.abs(D), np.angle(D)

    phi_unwrap = np.unwrap(phi, axis=1)
    delta_phi = np.diff(phi_unwrap, axis=1)


    magnitude, phase = librosa.magphase(D)
    
    magnitude = np.unwrap(magnitude, axis=1)
    
    fig, ax = plt.subplots(nrows=2, sharex=False, sharey=False)

    librosa.display.specshow(librosa.amplitude_to_db(S), y_axis='log', sr=sr, ax=ax[0])

    librosa.display.specshow(delta_phi, y_axis='log', x_axis='time', sr=sr, ax=ax[1])
    
    ax[0].set_title("Magnitude")
    ax[1].set_title("Phase")

    plt.tight_layout();
    
    plt.savefig(args.output)
    plt.show()
