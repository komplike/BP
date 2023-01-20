import wave
import numpy as np
import matplotlib.pyplot as plt
import code.volume_VAD.Volume as vp
from pathlib import Path
import os


def findIndex(vol, thres):
    L = len(vol)
    ii = 0
    index = np.zeros(300, dtype=np.int16)
    for i in range(L - 1):
        if((vol[i] - thres) * (vol[i + 1] - thres) < 0):
            index[ii] = i
            ii = ii + 1
    return index[[0, -1]]


def readFile(p, f, e):
    """Reads a .wav file.

    Takes the path, and returns (audio data, sample rate and number of frames).
    """
    fw = wave.open(p + f + e, 'r')
    params = fw.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = fw.readframes(nframes)
    waveData = np.frombuffer(strData, dtype=np.int16)
    waveData = waveData * 1.0 / max(abs(waveData))  # normalization
    fw.close()
    return waveData, framerate, nframes


def readFile2(p):
    """Reads a .wav file.

    Takes the path, and returns (audio data, sample rate and number of frames).
    """
    fw = wave.open(p, 'r')
    params = fw.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = fw.readframes(nframes)
    waveData = np.frombuffer(strData, dtype=np.int16)
    waveData = waveData * 1.0 / max(abs(waveData))  # normalization
    fw.close()
    return waveData, framerate, nframes


def get_graph(p, f, s):
    """Plot difference between tresholds

    Returns graphs with treshold marks
    """
    waveData, framerate, nframes = readFile(p, f, s)
    frameSize = int(round(0.025 * framerate))
    overLap = int(round(0.015 * framerate))
    vol = vp.calVolume(waveData, frameSize, overLap)
    threshold1 = max(vol) * 0.10
    threshold2 = min(vol) * 10.0
    threshold3 = max(vol) * 0.05 + min(vol) * 5.0
    time = np.arange(0, nframes) * (1.0 / framerate)
    vols = np.arange(0, len(vol)) * (nframes * 1.0 / len(vol) / framerate)
    index1 = findIndex(vol, threshold1) * (nframes * 1.0 / len(vol) / framerate)
    index2 = findIndex(vol, threshold2) * (nframes * 1.0 / len(vol) / framerate)
    index3 = findIndex(vol, threshold3) * (nframes * 1.0 / len(vol) / framerate)
    end = nframes * (1.0 / framerate)

    plt.subplot(211)
    plt.title("VAD01 using volume")
    plt.plot(time, waveData, color="black")
    plt.plot([index1, index1], [-1, 1], '-r')
    plt.plot([index2, index2], [-1, 1], '-g')
    plt.plot([index3, index3], [-1, 1], '-b')
    plt.ylabel('Amplitude')

    plt.subplot(212)
    plt.plot(vols, vol, color="black")
    plt.plot([0, end], [threshold1, threshold1], '-r', label="threshold 1")
    plt.plot([0, end], [threshold2, threshold2], '-g', label="threshold 2")
    plt.plot([0, end], [threshold3, threshold3], '-b', label="threshold 3")
    plt.legend()
    plt.ylabel('Volume(absSum)')
    plt.xlabel('time(seconds)')
    plt.savefig("VAD01")
    plt.show()


def vad(sens):
    """Process wav files

    available modifications: th = threshold1 / threshold2 / threshold3
    """
    paths = Path('/home/komplike/bp/nahravky').glob('**/TSK7/*.wav')
    for path in paths:
        # because path is object not string
        path_in_str = str(path)
        waveData, framerate, nframes = readFile2(path_in_str)
        frameSize = int(round(0.025 * framerate))
        overLap = int(round(0.015 * framerate))
        frameHop = frameSize - overLap
        vol = vp.calVolume(waveData, frameSize, overLap)
        threshold1 = float(max(vol) * sens)
        threshold2 = min(vol) * sens * 100
        threshold3 = max(vol) * sens / 2 + min(vol) * sens * 50

        th = threshold3
        VAD = 0
        cnt_pataka = 0
        cnt = 0
        f = open("/home/komplike/bp/vysledky/" + os.path.splitext(os.path.basename(path_in_str))[0] + ".lab", "w+")
        for i in vol:
            cnt += 1
            # print("[" + str(cnt) + "]")
            E = float(i[0])
            if E > th and VAD == 0:
                t_stamp_b = frameHop / framerate * cnt
                # print("B: " + str(t_stamp_b))
                VAD = 1
            elif E < th and VAD == 1:
                t_stamp_e = frameHop / framerate * cnt
                VAD = 0
                if t_stamp_e - t_stamp_b > 0.025:
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
                # print(" E: " + str(t_stamp_e))
        f.close()


def vad1(p, file, s, sens, th_type):
    """Process single wav file

    available modifications: th = threshold1 / threshold2 / threshold3
    """
    waveData, framerate, nframes = readFile(p, file, s)
    frameSize = int(round(0.025 * framerate))
    overLap = int(round(0.015 * framerate))
    frameHop = frameSize - overLap
    vol = vp.calVolume(waveData, frameSize, overLap)

    if th_type == 1:
        threshold1 = float(max(vol) * sens)
        th = threshold1
    elif th_type == 2:
        threshold2 = min(vol) * sens * 100
        th = threshold2
    elif th_type == 3:
        threshold3 = max(vol) * sens / 2 + min(vol) * sens * 50
        th = threshold3

    VAD = 0
    cnt_pataka = 0
    cnt = 0
    f = open(file + ".lab", "w+")
    for i in vol:
        cnt += 1
        # print("[" + str(cnt) + "]")
        E = float(i[0])
        if E > th and VAD == 0:
            t_stamp_b = frameHop / framerate * cnt
            # print("B: " + str(t_stamp_b))
            VAD = 1
        elif E < th and VAD == 1:
            t_stamp_e = frameHop / framerate * cnt
            VAD = 0
            if t_stamp_e - t_stamp_b > 0.025:
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
            # print(" E: " + str(t_stamp_e))
    f.close()


if __name__ == '__main__':
    PATH = "/home/komplike/bp/nahravky/drive/HC/TSK7/"
    FILE_NAME = ""  # e.g. HC_F_01_TSK7
    SUFFIX = ".wav"
    GRAPH = 0
    sens = 0.10

    if GRAPH == 1:
        get_graph(PATH, FILE_NAME, SUFFIX)

    if FILE_NAME == "":
        vad(sens)
    else:
        vad1(PATH, FILE_NAME, SUFFIX, sens, th_type)
