import pyaudio
import numpy as np
from silero_vad import load_silero_vad
import torch
from scipy import signal

from utils import *       # local import
import sounddevice as sd
import soundfile as sf
# ------------------------ Keyboard Input -------------------

from pynput import keyboard


stop_playback = False
main_loop = True

def on_press(key):
    global stop_playback
    if key == keyboard.Key.alt_r:
        stop_playback = True


def on_release(key):
    if key == keyboard.Key.esc:
        global main_loop , stop_playback
        main_loop = False
        stop_playback = True
        print('ESC pressed, stopped keyboard listener ;')
        return False

keyboard_listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
    )

keyboard_listener.start()


# --------------------------- main --------------------------
vad_model = load_silero_vad()


p = pyaudio.PyAudio()


# list_microphones() # list all mic devices
# print(sd.query_devices()) # list all speaker devices

FORMAT = pyaudio.paInt16  
CHANNELS = 1
RATE = 16000 #  44100 # 48000 # 
CHUNK = 512 # 1413 # 1536 # 
mic_index = None # 3
speaker_index = None # 18

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=mic_index,  
                frames_per_buffer=CHUNK)




def resample_audio(audio_array, original_rate, target_rate):
    num_samples = int(len(audio_array) * target_rate / original_rate)
    resampled_audio = signal.resample(audio_array, num_samples)
    return resampled_audio


buffer_size = int(RATE / CHUNK)  # Number of chunks in 1 second
audio_buffer = []
is_recording = False
recorded_frames = np.array([], dtype=np.int16)
voice_detection_counter = 0
silence_counter = 0


sd.default.device = speaker_index


try:
    print(">>>>>>>>>>>>>>>.", end='\r')
    while main_loop:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32)

        if RATE != 16000:
            audio_array = resample_audio(audio_array, RATE, 16000)
            # print('------------------------------------>>>', audio_array.shape)

        speech_probability = vad_model(torch.from_numpy(audio_array / 32768.0), 16000).item()
        # print(f'probability: {speech_probability}', end='\r')

        if speech_probability > 0.5:
            voice_detection_counter += 1
            silence_counter = 0
            print('Listening ...', end='\r')
        else:
            voice_detection_counter = 0 
            silence_counter += 1
            print('Silent ...', end='\r')
            

        audio_buffer.append(audio_array)
        if len(audio_buffer) > buffer_size*2:
            audio_buffer.pop(0)


        if voice_detection_counter >= buffer_size and not is_recording:
            is_recording = True
            recorded_frames = np.concatenate(audio_buffer)

        if is_recording:
            recorded_frames = np.concatenate((recorded_frames, audio_array))

        if is_recording and silence_counter > buffer_size:
            is_recording = False
            print('Processing ...', end='\r')

            
            user_text = speech_to_text(recorded_frames, 16000)
            print('::', user_text)
            

            reply_text = llm_reply(user_text)

            print('>>', reply_text)

            tts_audio_data, tts_sample_rate = kokoro_tts(reply_text)

            sd.play(tts_audio_data, samplerate=tts_sample_rate)
            
            while sd.get_stream().active:
                if stop_playback:
                    stop_playback = False
                    sd.stop() # stop speaking
                    break
                sd.sleep(500)


            # -- Reset variables --
            recorded_frames = np.array([], dtype=np.int16)
            audio_buffer = []
            voice_detection_counter = 0 
            print('----------------------------------------------------------------------<<<\n')
            


except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    if keyboard_listener:
        keyboard_listener.stop()




