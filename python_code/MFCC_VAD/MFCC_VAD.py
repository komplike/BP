from scipy import fftpack
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def add_eps(x):
    x[np.where(x == 0)] = np.finfo(dtype=x.dtype).eps
    return x


def preemphasis(seq, coeff):
    return np.append(seq[0], seq[1:] - coeff * seq[:-1])


# http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
def freq_to_mel(freq):
    return 1125.0 * np.lib.scimath.log(1.0 + freq / 700.0)


def mel_to_freq(mel):
    return 700.0 * (np.exp(mel / 1125.0) - 1.0)


def iter_bin(out, curr_bin, next_bins, backward=False):
    next_bin = next_bins[np.where(next_bins > curr_bin)][0]
    if backward:
        sign = -1
        bias = next_bin
    else:
        sign = 1
        bias = curr_bin
    for f in range(int(curr_bin), int(next_bin)):
        out[f] = sign * (f - bias) / (next_bin - curr_bin)


def mel_filterbank(num_bank, num_freq, sample_freq, low_freq, high_freq):
    num_fft = (num_freq - 1) * 2
    low_mel = freq_to_mel(low_freq)
    high_mel = freq_to_mel(high_freq)
    banks = np.linspace(low_mel, high_mel, num_bank + 2)
    bins = np.floor((num_fft + 1) * mel_to_freq(banks) / sample_freq)
    out = np.zeros((num_bank, num_fft // 2 + 1))
    for b in range(num_bank):
        iter_bin(out[b], bins[b], bins[b + 1:])
        iter_bin(out[b], bins[b + 1], bins[b + 2:], backward=True)
    return out


def main(data):
    filename = "HC_F_01_TSK7"
    data = "/home/komplike/bp/nahravky/drive/HC/TSK7/" + filename + ".wav"

    # config is based on Kaldi compute-mfcc-feats

    # STFT conf
    frame_length = 25  # frame / msec
    frame_shift = 10   # frame / msec
    remove_dc_offset = True
    window_type = "hamming"

    # Fbank conf
    preemphasis_coeff = 0.97
    use_power = True  # else use magnitude
    high_freq = 0.0  # offset from Nyquist freq [Hz]
    low_freq = 20.0  # offset from 0 [Hz]
    num_mel_bins = 24  # (default 23)
    num_ceps = 13
    num_lifter = 22

    sample_freq, raw_seq = wavfile.read(data)
    assert raw_seq.ndim == 1  # assume mono
    seq = raw_seq.astype(np.float64)
    if remove_dc_offset:
        seq -= np.mean(seq)

    # STFT feat
    seq = preemphasis(seq, preemphasis_coeff)
    num_samples = sample_freq // 1000
    window = signal.get_window(window_type, frame_length * num_samples)
    mode = "psd" if use_power else "magnitude"
    f, t, spectrogram = signal.spectrogram(seq, sample_freq, window=window, noverlap=frame_shift * num_samples, mode=mode)

    # log-fbank feat
    banks = mel_filterbank(num_mel_bins, spectrogram.shape[0], sample_freq, low_freq, sample_freq // 2 - high_freq)
    fbank_spect = np.dot(banks, spectrogram)
    logfbank_spect = np.lib.scimath.log(add_eps(fbank_spect))

    # mfcc feat
    dct_feat = fftpack.dct(logfbank_spect, type=2, axis=0, norm="ortho")[:num_ceps]
    lifter = 1 + num_lifter / 2.0 * np.sin(np.pi * np.arange(num_ceps) / num_lifter)
    mfcc_feat = lifter[:, np.newaxis] * dct_feat

    def savefig(name):
        return plt.savefig(name + ".svg")

    # plt.plot(seq)
    # savefig("signal")
    # plt.matshow(spectrogram)
    # savefig("spectrogram")
    # plt.matshow(banks)
    # savefig("banks")
    # plt.matshow(logfbank_spect)
    # savefig("logfbank")
    plt.matshow(mfcc_feat)
    # count = 0
    # fil = open("MFCC_coeff", "+w")
    # for row in mfcc_feat:
    #     for i in row:
    #         if count >= 50:
    #             count = 0
    #             break
    #         fil.write("%5.1f   " % (i))
    #         count += 1

    #     fil.write("\n")

    # fil.close()
    # print(np.size(mfcc_feat))
    # exit()
    plt.xlabel('frames')
    plt.ylabel('coefficients')
    # savefig("mfccoef")
    plt.show()
    exit()
    f = open("/home/komplike/bp/" + filename + ".lab", "w+")
    # f = open("/home/komplike/bp/vysledky/" + os.path.splitext(os.path.basename(data))[0] + ".lab", "w")
    test_cnt = 0
    for row in mfcc_feat:
        cnt = 0
        cnt_pataka = 0
        VAD = 0
        if test_cnt == 1:
            f.close()
            # print("exit in mfcc_vad.py")
            break
        test_cnt += 1
        cnt = 0
        for col in row:
            # if cnt == 0:
            if col > 0 and VAD == 0:
                t_stamp_b = cnt * 0.015
                VAD = 1
            elif col < 0 and VAD == 1:
                t_stamp_e = cnt * 0.015
                if t_stamp_e - t_stamp_b > 0.06:
                    VAD = 0
                    cnt_pataka += 1
                    if cnt_pataka == 1:
                        f.write(f'{t_stamp_b:.4f} ')
                        f.write(f'{t_stamp_e:.4f} pa\n')
                    elif cnt_pataka == 2:
                        f.write(f'{t_stamp_b:.4f} ')
                        f.write(f'{t_stamp_e:.4f} ta\n')
                    elif cnt_pataka == 3:
                        f.write(f'{t_stamp_b:.4f} ')
                        f.write(f'{t_stamp_e:.4f} ka\n')
                        cnt_pataka = 0

            cnt += 1
        if VAD == 1:
            t_stamp_e = len(raw_seq) / sample_freq
            cnt_pataka += 1
            if cnt_pataka == 1:
                f.write(f'{t_stamp_b:.4f} ')
                f.write(f'{t_stamp_e:.4f} pa\n')
            elif cnt_pataka == 2:
                f.write(f'{t_stamp_b:.4f} ')
                f.write(f'{t_stamp_e:.4f} ta\n')
            elif cnt_pataka == 3:
                f.write(f'{t_stamp_b:.4f} ')
                f.write(f'{t_stamp_e:.4f} ka\n')
                cnt_pataka = 0
    f.close()


if __name__ == '__main__':
    # paths = Path('/home/komplike/bp/nahravky').glob('**/TSK7/*.wav')
    # for path in paths:
    #     main(str(path))
    main("")
