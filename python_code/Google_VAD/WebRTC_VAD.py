import collections
import contextlib
import wave
import webrtcvad
import os
from pathlib import Path


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames, file):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    cnt_pataka = 0
    cnt = 0
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                file.write('%.4f' % ring_buffer[0][0].timestamp + " ")
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                cnt_pataka += 1
                if cnt_pataka == 1:
                    file.write('%.4f' % (frame.timestamp + frame.duration) + " pa\n")
                elif cnt_pataka == 2:
                    file.write('%.4f' % (frame.timestamp + frame.duration) + " ta\n")
                elif cnt_pataka == 3:
                    file.write('%.4f' % (frame.timestamp + frame.duration) + " ka\n")
                    cnt_pataka = 0
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
        cnt += 1
    if triggered:
        cnt_pataka += 1
        if cnt_pataka == 1:
            file.write('%.4f' % (frame.timestamp + frame.duration) + " pa\n")
        elif cnt_pataka == 2:
            file.write('%.4f' % (frame.timestamp + frame.duration) + " ta\n")
        elif cnt_pataka == 3:
            file.write('%.4f' % (frame.timestamp + frame.duration) + " ka\n")
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def main2(p, f, s, mode, fl):
    """Main method, process 1 audio file and create lab file in output
    directory
    """
    audio, sample_rate = read_wave(p + f + s)
    vad = webrtcvad.Vad(mode)  # 0-3
    frame_duration_ms = fl
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    file = open(f + ".lab", "w+")
    segments = vad_collector(sample_rate, frame_duration_ms, frame_duration_ms, vad, frames, file)
    for i, segment in enumerate(segments):
        continue
    file.close()


def main():
    """Main method, process all audio files in directory
    and create lab file for each file in output directory
    """
    paths = Path('/home/komplike/bp/nahravky').glob('**/TSK7/*.wav')
    for path in paths:
        # because path is object not string
        path_in_str = str(path)
        audio, sample_rate = read_wave(path_in_str)
        vad = webrtcvad.Vad(3)  # 1-3
        frame_duration_ms = 20
        frames = frame_generator(frame_duration_ms, audio, sample_rate)
        frames = list(frames)
        file = open("/home/komplike/bp/vysledky/" + os.path.splitext(os.path.basename(path_in_str))[0] + ".lab", "w+")
        segments = vad_collector(sample_rate, frame_duration_ms, frame_duration_ms, vad, frames, file)
        for i, segment in enumerate(segments):
            continue
        file.close()


if __name__ == '__main__':
    """Process of the program, calls method main or main2
    if filename is given processing 1 file, otherwise all files in directory
    """
    PATH = "/home/komplike/bp/nahravky/drive/PD/TSK7/"
    FILE_NAME = ""  # e.g. HC_F_01_TSK7
    SUFFIX = ".wav"
    if FILE_NAME == "":
        main()
    else:
        main2(PATH, FILE_NAME, SUFFIX)
