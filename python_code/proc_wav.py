# All TSK7 wav files processing
# import used libraries
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from pydub import AudioSegment
import os
# energy based VAD


def vad_e(ffilename, fframes, fframe_length, fnum_frames, fframe_hop):
    E = 0
    Ef = 0
    for r in frames:
        temp = 0
        for c in r:
            temp += c**2
        Ef = temp / frame_length
        E += Ef
    E = E / num_frames
    treshold = int(round(E * 0.2))
    cnt = 0
    VAD = 0
    wCnt = 0
    for r in frames:
        cnt += 1
        temp = 0
        for c in r:
            temp += c**2
        Ef = temp / frame_length
        if Ef < treshold and VAD == 0:
            t_stamp_b = fframe_hop * cnt
            VAD = 1
        elif Ef > treshold and VAD == 1:
            t_stamp_e = fframe_hop * cnt
            VAD = 0
            newAudio = AudioSegment.from_wav(ffilename)
            newAudio = newAudio[t_stamp_b * 1000:t_stamp_e * 1000]
            newAudio.export("/home/komplike/bp/vysledky/wav_nonspeech/" + os.path.splitext(os.path.basename(ffilename))[0] + "_s_" + str(wCnt) + ".wav", format="wav")
            wCnt += 1


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

    vad_e(path_in_str, frames, frame_length, num_frames, frame_hop)
    print(path_in_str + ": DONE!")

# # single wav file processing
# # import used libraries
# import numpy as np
# from scipy.io import wavfile

# # energy based VAD


# def vad_e(ffilename, fframes, fframe_length, fnum_frames, fframe_hop):
#     E = 0
#     Ef = 0
#     for r in frames:
#         temp = 0
#         for c in r:
#             temp += c**2
#         Ef = temp / frame_length
#         E += Ef
#     E = E / num_frames
#     treshold = int(round(E * 0.2))
#     cnt = 0
#     cnt_pataka = 0
#     VAD = 0
#     f = open(ffilename + ".lab", "w+")
#     for r in frames:
#         cnt += 1
#         temp = 0
#         for c in r:
#             temp += c**2
#         Ef = temp / frame_length
#         if Ef > treshold and VAD == 0:
#             t_stamp_b = fframe_hop * cnt
#             VAD = 1
#         elif Ef < treshold and VAD == 1:
#             t_stamp_e = fframe_hop * cnt
#             VAD = 0
#             if t_stamp_e - t_stamp_b > 0.025:
#                 cnt_pataka += 1
#                 if cnt_pataka == 1:
#                     f.write(f'{t_stamp_b:.4f} ')
#                     f.write(f'{t_stamp_e:.4f} pa\n')
#                 elif cnt_pataka == 2:
#                     f.write(f'{t_stamp_b:.4f} ')
#                     f.write(f'{t_stamp_e:.4f} ta\n')
#                 elif cnt_pataka == 3:
#                     f.write(f'{t_stamp_b:.4f} ')
#                     f.write(f'{t_stamp_e:.4f} ka\n')
#                     cnt_pataka = 0

#     f.close()


# filename = "HC_F_01_TSK7"
# # read signal
# sample_rate, signal = wavfile.read("./nahravky/drive/HC/TSK7/" + filename + ".wav")
# # apply preemphasis filter
# pre_emphasis = 0.95
# emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
# # define frame properties
# frame_size = 0.025  # s
# frame_hop = 0.015  # s
# # sec2samples
# frame_length = frame_size * sample_rate
# frame_step = frame_hop * sample_rate

# signal_length = len(emphasized_signal)
# frame_length = int(round(frame_length))
# frame_step = int(round(frame_step))
# num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
# # add zero padding to have the same length of each frame
# pad_signal_length = num_frames * frame_step + frame_length
# z = np.zeros((pad_signal_length - signal_length))
# pad_signal = np.append(emphasized_signal, z)
# # index frames
# indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
# # change type to int and disable reallocation
# frames = pad_signal[indices.astype(np.int32, copy=False)]
# # apply hamming window to each frame
# frames *= np.hamming(frame_length)

# vad_e(filename, frames, frame_length, num_frames, frame_hop)
