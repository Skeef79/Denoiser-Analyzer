import os
import numpy as np
import soundfile as sf
import subprocess
import glob
import librosa
import random
import tempfile

def audioread(path, start=0, stop=None):
    """returns audio, sample_rate """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError(f"[{path}] does not exist!")
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:
        print('WARNING: Audio type not supported')

    if len(audio.shape) > 1: #is it is multi-channel
        audio = audio.T
        audio = audio.sum(axis=0)/audio.shape[0]
        
    return audio, sample_rate

def audiowrite(destpath, audio, sample_rate=16000):
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return

