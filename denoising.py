from pyroomacoustics.denoise.spectral_subtraction import apply_spectral_sub
from pyroomacoustics.denoise.iterative_wiener import apply_iterative_wiener
from pyroomacoustics.utilities import normalize_pwr, normalize
from audio_utils import audioread, audiowrite
import os
from tqdm import tqdm


def denoise_demucs(model_type, noisy_dir, out_dir):
    """
    model_type : dns48, dns64, master64 (str)
    noisy_dir : directory of noisy files to enhance (str)
    out_dir: directory, where to store resulsts (str)
    
    """
    interpreter = 'python'
    command = f'{interpreter} -m denoiser.enhance --{model_type} --noisy_dir={noisy_dir} --out_dir={out_dir}'
    os.system(command)
    

def denoise_spectral_sub(cfg):
    noisy_paths = [os.path.join(cfg.results_dir, 'noisy', file) for file in os.listdir(os.path.join(cfg.results_dir,'noisy'))]
    for file in tqdm(noisy_paths):
        signal, _ = audioread(file)
        result = apply_spectral_sub(signal,db_reduc = 5)    
        result = normalize(normalize_pwr(result, signal))
        audiowrite(os.path.join(cfg.results_dir, 'enhanced_spectral_sub',file.split('/')[-1]), result, 16000)
                   

def denoise_wiener(cfg):
    noisy_paths = [os.path.join(cfg.results_dir, 'noisy', file) for file in os.listdir(os.path.join(cfg.results_dir,'noisy'))]
    for file in tqdm(noisy_paths):
        signal, _ = audioread(file)
        result = apply_iterative_wiener(signal,frame_len=1024, lpc_order=10)
        result = normalize_pwr(result, signal)
        audiowrite(os.path.join(cfg.results_dir, 'enhanced_wiener',file.split('/')[-1]), result, 16000)
    

    