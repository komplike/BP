# import used libraries
import numpy as np
from scipy.io import wavfile
import os
from pathlib import Path


def vad_e(ffilename, fframes, fframe_length, fnum_frames, fframe_hop, sens):
    """Method processing wav file
    Calculates total energy and set treshold to 20% of it
    Stores time stamps of beginning and end of voiced segments in .lab file
    """
    E = 0
    Ef = 0
    # calculates energy of audio
    for r in frames:
        temp = 0
        for c in r:
            temp += c**2
        Ef = temp / frame_length
        E += Ef
    E = E / num_frames
    # set treshold
    treshold = int(round(E * sens))
    cnt = 0
    cnt_pataka = 0
    VAD = 0
    f = open("/home/komplike/bp/vysledky/" + os.path.splitext(os.path.basename(ffilename))[0] + ".lab", "w")
    # iterate frames and evaluating begin and end of voiced parts
    for r in frames:
        cnt += 1
        temp = 0
        for c in r:
            temp += c**2
        Ef = temp / frame_length
        if Ef > treshold and VAD == 0:
            t_stamp_b = fframe_hop * cnt
            VAD = 1
        elif Ef < treshold and VAD == 1:
            t_stamp_e = fframe_hop * cnt
            if t_stamp_e - t_stamp_b < 0.025:
                VAD = 0
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

    f.close()


def vad_e2(ffilename, frames, frame_length, num_frames, fframe_hop, sens):
    """mirrored vad_e method for 1 audio file only
    """
    E = 0
    Ef = 0
    for r in frames:
        temp = 0
        for c in r:
            temp += c**2
        Ef = temp / frame_length
        E += Ef
    E = E / num_frames
    treshold = int(round(E * sens))
    cnt = 0
    cnt_pataka = 0
    VAD = 0
    f = open(ffilename + ".lab", "w+")
    for r in frames:
        cnt += 1
        temp = 0
        for c in r:
            temp += c**2
        Ef = temp / frame_length
        if Ef > treshold and VAD == 0:
            t_stamp_b = fframe_hop * cnt
            VAD = 1
        elif Ef < treshold and VAD == 1:
            t_stamp_e = fframe_hop * cnt
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

    f.close()


def vad_e2x(ffilename, frames, frame_length, num_frames, fframe_hop, sens):
    """mirrored vad_e method for 1 audio file only
    """
    E = 0
    Ef = 0
    for r in frames:
        temp = 0
        for c in r:
            temp += c**2
        Ef = temp / frame_length
        E += Ef
    E = E / num_frames
    treshold = int(round(E * sens))
    cnt = 0
    cnt_pataka = 0
    VAD = 0
    f = open(ffilename + ".lab", "w+")
    for r in frames:
        cnt += 1
        temp = 0
        for c in r:
            temp += c**2
        Ef = temp / frame_length
        if Ef >= treshold and VAD == 0:
            t_stamp_b = fframe_hop * cnt
            VAD = 1
        elif Ef < treshold and VAD == 1:
            t_stamp_e = fframe_hop * cnt
            VAD = 0
            if 1:
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
    if VAD == 1:
        t_stamp_e = fframe_hop * cnt
        f.write(f'{t_stamp_b:.4f} ')
        f.write(f'{t_stamp_e:.4f}\n')
    f.close()


def one_vad_e(path, FILE_NAME, sens):
    # read signal
    sample_rate, signal = wavfile.read(path + FILE_NAME + ".wav")
    # apply preemphasis filter
    pre_emphasis = 0.95
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    # define frame properties
    frame_size = 0.025  # s
    frame_hop = 0.015  # s
    # sec2samples
    frame_length = frame_size * sample_rate
    frame_step = frame_hop * sample_rate

    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # add zero padding to have the same length of each frame
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    # index frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    # change type to int and disable reallocation
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # apply hamming window to each frame
    frames *= np.hamming(frame_length)

    vad_e2x(FILE_NAME, frames, frame_length, num_frames, frame_hop, sens)


if __name__ == '__main__':
    """main process calls method vad_E or vad_e2
    if filename is given processing 1 file, otherwise all files in directory
    """
    FILE_NAME = ""  # e.g. "HC_F_01_TSK7"
    sens = 0.015
    one_vad_e("/home/komplike/bp/nahravky/drive/PD/TSK7/", "PD_F_01_TSK7", sens)
    exit()

    if FILE_NAME == "":
        paths = Path('/home/komplike/bp/nahravky').glob('**/TSK7/*.wav')
        for path in paths:
            # because path is object not string
            path_in_str = str(path)

            # read signal
            sample_rate, signal = wavfile.read(path_in_str)
            # apply preemphasis filter
            pre_emphasis = 0.95
            emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
            # define frame properties
            frame_size = 0.025  # s
            frame_hop = 0.015  # s
            # sec2samples
            frame_length = frame_size * sample_rate
            frame_step = frame_hop * sample_rate

            signal_length = len(emphasized_signal)
            frame_length = int(round(frame_length))
            frame_step = int(round(frame_step))
            num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
            # add zero padding to have the same length of each frame
            pad_signal_length = num_frames * frame_step + frame_length
            z = np.zeros((pad_signal_length - signal_length))
            pad_signal = np.append(emphasized_signal, z)
            # index frames
            indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
            # change type to int and disable reallocation
            frames = pad_signal[indices.astype(np.int32, copy=False)]
            # apply hamming window to each frame
            frames *= np.hamming(frame_length)

            vad_e(path_in_str, frames, frame_length, num_frames, frame_hop, sens)
            print(path_in_str + ": DONE!")
    else:
        # read signal
        sample_rate, signal = wavfile.read("./nahravky/drive/HC/TSK7/" + FILE_NAME + ".wav")
        # apply preemphasis filter
        pre_emphasis = 0.95
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        # define frame properties
        frame_size = 0.025  # s
        frame_hop = 0.015  # s
        # sec2samples
        frame_length = frame_size * sample_rate
        frame_step = frame_hop * sample_rate

        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        # add zero padding to have the same length of each frame
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)
        # index frames
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        # change type to int and disable reallocation
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        # apply hamming window to each frame
        frames *= np.hamming(frame_length)

        vad_e2(FILE_NAME, frames, frame_length, num_frames, frame_hop, sens)
