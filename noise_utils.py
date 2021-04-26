import numpy as np
import random
import os

EPS = np.finfo(float).eps

def normalize(signal, target_level=-25):
    rms = (signal ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    signal = signal * scalar
    return signal

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)

def _apply_noise(clean, noise, snr):
    
    target_level = -25
    clipping_threshold=0.99
    
    clean = clean/(max(abs(clean))+EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean**2).mean()**0.5
    
    noise = noise/(max(abs(noise))+EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise**2).mean()**0.5

    
    alpha = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
    noisenewlevel = noise * alpha
    
    noisyspeech = clean + noisenewlevel
    
    noisy_rms_level = -25
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech))/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel
        clean = clean/noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel/noisyspeech_maxamplevel
        noisy_rms_level = int(20*np.log10(scalarnoisy/noisyspeech_maxamplevel*(rmsnoisy+EPS)))

    return noisyspeech


def get_white_noise(STD, signal_length):
    return np.random.normal(0, STD, signal_length)


def apply_noise(signal, noise, SNR):
    if (len(signal)<=len(noise)):
        noise = noise[:len(signal)].copy()
        return _apply_noise(signal, noise, SNR)
    else:
        noise = np.append(noise, np.zeros(len(clean)-len(noise)))
        return _apply_noise(signal, noise, SNR)
    
    
    
    