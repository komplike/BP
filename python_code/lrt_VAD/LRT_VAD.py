# -*- coding: utf-8 -*-
"""
likelihood ratio test
Created on May  1 20:43:28 2021
@author: eesungkim edited by Roman Santa for semestrl thesis

A voice activity detector applied a statistical model has been made in [2],
where the decision rule is derived from the likelihood ratio test (LRT)
by estimating unknown parameters using the decision-directed method.
Hang-over scheme based on the hidden Markov model (HMM) are applied for smoothing.

Reference:
[1] Project Repository https://github.com/eesungkim/Voice_Activity_Detector
[2]  Y. Ephraim and D. Malah, "Speech enhancement using a minimum-mean square error
short-time spectral amplitude estimator,"
IEEE Trans Acoustics Speech and Signal Processing, VOL. 32(6):1109-1121, Dec 1984.
"""

import numpy as np
import scipy.io.wavfile as wav
import math
import os
from pathlib import Path
from code.lrt_VAD.LRT_estnoise_ms import *


def VAD(f, signal, sr, nFFT=512, win_length=0.025, hop_length=0.01, theshold=0.7):
    """Voice Activity Detector
    Parameters
    ----------
    signal      : audio time series
    sr          : sampling rate of `signal`
    nFFT        : length of the FFT window
    win_length  : window size in sec
    hop_length  : hop size in sec

    Returns
    -------
    probRatio   : frame-based voice activity probability sequence
    """
    signal = signal.astype('float')

    maxPosteriorSNR = 1000
    minPosteriorSNR = 0.0001

    win_length_sample = round(win_length * sr)
    hop_length_sample = round(hop_length * sr)

    # the variance of the speech; lambda_x(k)
    _stft = stft(signal, n_fft=nFFT, win_length=win_length_sample, hop_length=hop_length_sample)
    pSpectrum = np.abs(_stft) ** 2

    # estimate the variance of the noise using minimum statistics noise PSD estimation ; lambda_d(k).
    estNoise = estnoisem(pSpectrum, hop_length)
    estNoise = estNoise

    aPosterioriSNR = pSpectrum / estNoise
    aPosterioriSNR = aPosterioriSNR
    aPosterioriSNR[aPosterioriSNR > maxPosteriorSNR] = maxPosteriorSNR
    aPosterioriSNR[aPosterioriSNR < minPosteriorSNR] = minPosteriorSNR

    a01 = hop_length / 0.05     # a01=P(signallence->speech)  hop_length/mean signallence length (50 ms)
    a00 = 1 - a01               # a00=P(signallence->signallence)
    a10 = hop_length / 0.1      # a10=P(speech->signallence) hop/mean talkspurt length (100 ms)
    a11 = 1 - a10               # a11=P(speech->speech)

    b01 = a01 / a00
    b10 = a11 - a10 * a01 / a00

    smoothFactorDD = 0.99
    previousGainedaPosSNR = 1
    (nFrames, nFFT2) = pSpectrum.shape
    probRatio = np.zeros((nFrames, 1))
    logGamma_frame = 0
    VAD = 0
    cnt_pataka = 0
    for i in range(nFrames):
        aPosterioriSNR_frame = aPosterioriSNR[i, :]

        # operator [2](52)
        oper = aPosterioriSNR_frame - 1
        oper[oper < 0] = 0
        smoothed_a_priori_SNR = smoothFactorDD * previousGainedaPosSNR + (1 - smoothFactorDD) * oper

        # V for MMSE estimate ([2](8))
        V = 0.1 * smoothed_a_priori_SNR * aPosterioriSNR_frame / (1 + smoothed_a_priori_SNR)

        # geometric mean of log likelihood ratios for individual frequency band  [1](4)
        logLRforFreqBins = 2 * V - np.log(smoothed_a_priori_SNR + 1)
        # logLRforFreqBins=np.exp(smoothed_a_priori_SNR*aPosterioriSNR_frame/(1+smoothed_a_priori_SNR))/(1+smoothed_a_priori_SNR)
        gMeanLogLRT = np.mean(logLRforFreqBins)
        logGamma_frame = np.log(a10 / a01) + gMeanLogLRT + np.log(b01 + b10 / (a10 + a00 * np.exp(-logGamma_frame)))
        probRatio[i] = 1 / (1 + np.exp(-logGamma_frame))

        # Calculate Gain function which results from the MMSE [2](7).
        gain = (math.gamma(1.5) * np.sqrt(V)) / aPosterioriSNR_frame * np.exp(-1 * V / 2) * ((1 + V) * bessel(0, V / 2) + V * bessel(1, V / 2))

        previousGainedaPosSNR = (gain**2) * aPosterioriSNR_frame
        if probRatio[i] > theshold and not VAD:
            VAD = 1
            t_stamp_b = hop_length * i

        elif probRatio[i] <= theshold and VAD:

            t_stamp_e = hop_length * i
            if t_stamp_e - t_stamp_b > 0.08:
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
        probRatio[probRatio > theshold] = 1
        probRatio[probRatio <= theshold] = 0
    if VAD == 1:
        t_stamp_e = hop_length * i
        f.write(f'{t_stamp_b:.4f} ')
        f.write(f'{t_stamp_e:.4f}\n')
    return probRatio


def main():
    """Method processing all wav files

    Returns .lab files with time stamps
    """
    paths = Path('/home/komplike/bp/nahravky').glob('**/TSK7/*.wav')
    for path in paths:
        # because path is object not string
        path_in_str = str(path)
        file = open("/home/komplike/bp/vysledky/" + os.path.splitext(os.path.basename(path_in_str))[0] + ".lab", "w+")
        (sr, signal) = wav.read(path_in_str)
        VAD(file, signal, sr, nFFT=512, win_length=0.025, hop_length=0.015, theshold=0.8)
        file.close()


def main2(p, f, s, th):
    """Method processing single wav file

    Returns .lab file with time stamps
    """
    (sr, signal) = wav.read(p + f + s)
    file = open(f + ".lab", "w+")
    print("one")
    VAD(file, signal, sr, nFFT=512, win_length=0.025, hop_length=0.015, theshold=th)
    file.close()


if __name__ == '__main__':
    """Process of the program, calls method main or main2
    if filename is given processing 1 file, otherwise all files in directory
    """
    PATH = "/home/komplike/bp/nahravky/drive/HC/TSK7/"
    FILE_NAME = ""  # e.g. HC_F_01_TSK7
    SUFFIX = ".wav"
    if FILE_NAME == "":
        main()
    else:
        main2(PATH, FILE_NAME, SUFFIX)
