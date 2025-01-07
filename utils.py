
# ------------------------------ Imports -----------------------------
import sounddevice as sd
import torch
import sys

import whisper
from librosa import resample

import numpy as np

from langchain_community.llms import Ollama
# ---------------------------- LLM -------------------------------------------


message_history = []



llm_model = Ollama(model="llama3.1:8b")


def llm_reply(message):
    max_msg_len = 10*3 # 15 msg stored
    message_history.append({"role": "user", "content": message})
    system_msg = """You are a helpful AI assistant, you are been implemented with a voice assistant system, so make sure to autocorrect words based on context.
    1. You will be asked to generate sets of questions, and ask one at a time, make sure user answerd the question, and you do the feedback, then ask next question.
    2. Keep your response as short as possible.
    You are question asking and feedback giving ai assistant, your name is quizo.
    """

    messages=[{"role": "system", "content": system_msg}] + message_history[-max_msg_len:]

    reply = llm_model.invoke(messages)

    message_history.append({"role": "assistant", "content": reply})

    return reply


# --------------------------------- TTS ---------------------------------------
import requests, os

library_path = 'Kokoro-82M'
sys.path.append(library_path)

from models import build_model # local import
from kokoro import generate, generate_large_text # using custom made function;

# ---- Download pth file --- for tts -----

file_name = "kokoro-v0_19.pth"
file_path = os.path.join(library_path, file_name)
url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.pth?download=true"

if not os.path.exists(file_path):
    print('Please wait, downloading tts model file ....', end='\r')
    with open(file_path, 'wb') as file:
        file.write(requests.get(url).content)
    print(f"Downloaded {file_name} to {library_path}.", end='\r')
# ----------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model(f'{library_path}/kokoro-v0_19.pth', device)
VOICE_NAME = [
    'af', # Default voice is a 50-50 mix of Bella & Sarah
    'af_bella', 'af_sarah', 'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky',
][0]
VOICEPACK = torch.load(f'{library_path}/voices/{VOICE_NAME}.pt', weights_only=True).to(device)
print(f'Loaded voice: {VOICE_NAME}', end='\r')

def kokoro_tts(input_text):
    input_text = input_text.replace('*', '') # remving **
    audio, out_ps = generate_large_text(MODEL, input_text, VOICEPACK, lang=VOICE_NAME[0])
    return audio, 24000
    # sd.play(audio, 24000)
    # sd.wait()  

# ------------------------------- STT ------------------------------------------------


whisper_model = whisper.load_model("large-v3-turbo")

def speech_to_text(audio_data, sample_rate):
    
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.squeeze().numpy()
    
    # sf.write('recorded_audio.wav', audio_data, sample_rate)

    audio_data = audio_data.astype(np.float32)


    if sample_rate != 16000:
        audio_data = resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000


    audio_data = audio_data.astype(np.float32)
    result = whisper_model.transcribe(audio_data, language='en', )
    return result["text"]



# ------------------ List all mic ----

import pyaudio

def list_microphones():
    p = pyaudio.PyAudio()
    
    print("Available microphones:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Index: {i}, Name: {device_info.get('name')}")

    p.terminate()



