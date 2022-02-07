import soundfile as sf
from sklearn.decomposition import FastICA
import numpy as np
from matplotlib import pyplot as plt
import sys
import os


def read_audio(path1, path2):
    observed1, o1fs = sf.read(path1)
    print('observed1: length = {} and sampling freq : = {} '.format(len(observed1), o1fs))

    observed2, o2fs = sf.read(path2)
    print('observed2: length = {} and sampling freq : = {}  '.format(len(observed2), o2fs))
    print("Shape of Observed1 and Observed2")
    print(observed1.shape, observed2.shape)

    return observed1, o1fs, observed2, o2fs


def convert_single_channel(observed1, observed2):
    if len(observed1) > len(observed2):
        observed1 = observed1[:len(observed2)]
    else:
        observed2 = observed2[:len(observed1)]

    print('After Slicing Length & channels\nObserved1 = {}, Observed2 = {}'.format(len(observed1), len(observed2)))

    # print(voice.shape, music.shape)
    a, b = observed1.T  # .mpeg has 2 channels. so taking the average
    observed1 = (a + b) / 2

    c, d = observed2.T  # .mpeg has 2 channels. so taking the average
    observed2 = (c + d) / 2
    print(observed1.shape, observed2.shape)

    return observed1, observed2


def plot_input_signals(observed1, observed2, path):
    plt.subplot(3, 1, 1)
    plt.scatter(observed1, observed2)
    plt.title('Correlation B/w Observed1 and Observed2')

    plt.subplot(3, 1, 2)
    x = np.arange(len(observed1))
    plt.plot(x, observed1)
    plt.title('Observed1')

    plt.subplot(3, 1, 3)
    plt.plot(x, observed2)
    plt.title('Observed2')
    plt.tight_layout()
    plt.savefig(path+'/observed_signals.png')
    plt.show()


def perform_ica(observed1, observed2):
    combinedSignals = np.c_[observed1, observed2]

    ica = FastICA(n_components=2)
    S_ = ica.fit_transform(combinedSignals)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    print('MIXING MATRIX A\n', A_)

    return S_, A_


def plot_output_signals(S_, path):
    xs = np.arange(len(S_))
    plt.plot(xs, S_)
    plt.title('Independent Sources')
    plt.savefig(path + '/independent_sources.png')
    plt.show()


def make_directory():
    directory = "ICA Components"
    current_dir = os.curdir
    path = os.path.join(current_dir, directory)
    try:
        os.mkdir(path)
    except:
        pass

    return path


def write_components(S_, o1fs, o2fs, path):
    comp1, comp2 = S_.T

    sf.write(path+'/out1_comp.wav', comp1, o1fs)
    sf.write(path+'/out2_comp.wav', comp2, o2fs)


if __name__ == '__main__':
    print('Usage: python ica.py <observed1.wav> <observed2.wav>')
    path = make_directory()
    if len(sys.argv) != 3:
        print("Incorrect Usage!!!\nTry: python ica.py <observed1.wav> <observed2.wav>")
    else:
        path1 = sys.argv[1]
        path2 = sys.argv[2]
        observed1, o1fs, observed2, o2fs = read_audio(path1, path2)
        observed1, observed2 = convert_single_channel(observed1, observed2)
        plot_input_signals(observed1, observed2, path)
        S_, A_ = perform_ica(observed1, observed2)
        plot_output_signals(S_, path)
        write_components(S_, o1fs, o2fs, path)
