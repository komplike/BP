# from scipy.io import wavfile
# import scipy
# from glob import glob

# dir_name = "vysledky/wav_speech"
# fs, c = wavfile.read("vysledky/wav_speech/HC_F_02_TSK7_s_0.wav")
# # for file in glob(dir_name + '/*.wav'):
# #     fs, a = wavfile.read(file)
# #     res = scipy.vstack((c, a))
# fs, c = wavfile.read("vysledky/wav_speech/HC_F_02_TSK7_s_0.wav")
# fs, a = wavfile.read("vysledky/wav_speech/HC_F_02_TSK7_s_1.wav")
# res = scipy.vstack((c, a))
# wavfile.write("vysledky/wav_speech/all_speech.wav", fs, res)
from glob import glob
from pydub import AudioSegment
dir_name = "vysledky/wav_nonspeech"
combined_sounds = AudioSegment.from_wav("vysledky/wav_nonspeech/HC_F_02_TSK7_s_0.wav")
for file in glob(dir_name + '/*.wav'):
    sound = AudioSegment.from_wav(file)
    combined_sounds += sound

combined_sounds.export("vysledky/wav_speech/all_nonspeech.wav", format="wav")
