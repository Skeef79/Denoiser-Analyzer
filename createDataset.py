from config import Config
import os
from shutil import copyfile
from tqdm import tqdm
from audio_utils import audioread, audiowrite
import json
from noise_utils import apply_noise

def createDataset(cfg):

    if not os.path.exists(cfg.results_dir):
        os.mkdir(cfg.results_dir)

    if not os.path.exists(os.path.join(cfg.results_dir,'clean')):
        os.mkdir(os.path.join(cfg.results_dir,'clean'))

    if not os.path.exists(os.path.join(cfg.results_dir,'noisy')):
        os.mkdir(os.path.join(cfg.results_dir,'noisy'))
    
    
    copyfile(cfg.config_path, os.path.join(cfg.results_dir,'config.json'))

    new_transcriptions = {}
    for idx, clean_file in tqdm(enumerate(cfg.clean_paths)):
        random_noise_file = cfg.getRandomNoiseFile()

        clean, sr = audioread(clean_file)
        noise, sr = audioread(random_noise_file)

        noisy = apply_noise(clean, noise, cfg.snr)

        audiowrite(os.path.join(cfg.results_dir, 'clean', f'{idx}.wav'),clean)
        audiowrite(os.path.join(cfg.results_dir, 'noisy', f'{idx}.wav'), noisy)

        new_transcriptions[idx] = cfg.transcriptions[clean_file]
    
    with open(os.path.join(cfg.results_dir,'transcriptions.json'),'w') as f:
        json.dump(new_transcriptions,f)

