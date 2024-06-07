import json
import torch
import torchaudio
import librosa
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_local_model():
    with open('model_config.json') as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config) # might need to import this
    model.load_state_dict(load_ckpt_state_dict('model.ckpt'))
    return model, model_config

def load_audio_input(path='input.wav', model_config=None, mono=False):
    audio, sr = librosa.load(path, sr=model_config["sample_rate"], mono=mono)
    audio = torch.from_numpy(audio).to(torch.float32)
    audio = (sr, audio) # Reshape tuple to fit model input
    return audio

def load_model():
    model, model_config = load_local_model()
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model = model.to(device)
    return model, model_config, sample_rate, sample_size

def generate_audio(output_name, input_audio=None, init_noise_level=1.0):
    # if receiving int bit size error: stable_audio_tools\inference\generation.py line 138 -> 2**31
    output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device,
    init_audio=input_audio, # A tuple of (sample_rate, audio)
    init_noise_level=init_noise_level
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to file
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save(f"output/{output_name}.wav", output, sample_rate)

if __name__ == '__main__':
    model, model_config, sample_rate, sample_size = load_model()

    # use "sample in a dry recording studio" to have it actually be samples LOL

    # Input Params
    USING_SINGLE_INPUT = False
    USING_GLITCHIFY = True
    PROMPT = 'electronic cinematic percussion loop glitch'
    NOISE_STRENGTH = 50.0
    NUM_GENERATIONS = 20 # This is also the number of input audio files if using GLITCHIFY

    conditioning = [{
    "prompt": PROMPT,
    "seconds_start": 0, 
    "seconds_total": 30
    }]

    if USING_SINGLE_INPUT:
        input_audio = load_audio_input(path='input.wav', model_config=model_config)
        for i in range(NUM_GENERATIONS):
            generate_audio(f'output_{i+1}', input_audio=input_audio, init_noise_level=NOISE_STRENGTH)
    elif USING_GLITCHIFY:
        for i in range(NUM_GENERATIONS):
            input_audio = load_audio_input(path=f'input/output_{i+1}.wav', model_config=model_config)
            generate_audio(f'output_{i+1}', input_audio=input_audio, init_noise_level=NOISE_STRENGTH)
    else:
        for i in range(NUM_GENERATIONS):
            generate_audio(f'output_{i+1}', input_audio=None)

