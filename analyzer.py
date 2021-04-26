from audio_utils import audioread, audiowrite
from config import Config
from text_utils import normalizeString
from createDataset import createDataset
from metrics import get_pesq, get_stoi, get_wer
from denoising import denoise_demucs, denoise_spectral_sub
import json
import os

import nemo
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl

from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf


config_path = 'config.json'
print("Config preparation started")
cfg = Config(config_path)
print("Config prepararation done")

print("Dataset creation started")
createDataset(cfg)
print("Dataset creation done")

if not os.path.exists(os.path.join(cfg.results_dir,'enhanced_demucs')):
    os.mkdir(os.path.join(cfg.results_dir, 'enhanced_demucs'))

if not os.path.exists(os.path.join(cfg.results_dir,'enhanced_spectral_sub')):
    os.mkdir(os.path.join(cfg.results_dir, 'enhanced_spectral_sub'))
    
noisy_dir = os.path.join(cfg.results_dir, 'noisy')
out_dir = os.path.join(cfg.results_dir, 'enhanced_demucs')

print("Denoising started")
denoise_demucs('master64', noisy_dir, out_dir)
denoise_spectral_sub(cfg)
print("Denoising done")

def getSounds(cfg):
    
    clean = []
    noisy = []
    enhanced_demucs = []
    enhanced_spectral_sub = []
    #enhanced_wiener = []
    
    clean_paths = [os.path.join(cfg.results_dir, 'clean', file) for file in os.listdir(os.path.join(cfg.results_dir,'clean'))]
    noisy_paths = [os.path.join(cfg.results_dir, 'noisy', file) for file in os.listdir(os.path.join(cfg.results_dir,'noisy'))]
    enhanced_demucs_paths = [os.path.join(cfg.results_dir, 'enhanced_demucs', file) for file in os.listdir(os.path.join(cfg.results_dir,'enhanced_demucs'))]
    enhanced_spectral_sub_paths = [os.path.join(cfg.results_dir, 'enhanced_spectral_sub', file) for file in os.listdir(os.path.join(cfg.results_dir,'enhanced_spectral_sub'))]
   
    clean_paths.sort()
    noisy_paths.sort()
    enhanced_demucs_paths.sort()
    enhanced_spectral_sub_paths.sort()
    
    for file in clean_paths:
        signal, _ = audioread(file)
        clean.append(signal)
    
    for file in noisy_paths:
        signal, _ = audioread(file)
        noisy.append(signal)
        
    for file in enhanced_demucs_paths:
        signal, _ = audioread(file)
        enhanced_demucs.append(signal)
        
    for file in enhanced_spectral_sub_paths:
        signal, _ = audioread(file)
        enhanced_spectral_sub.append(signal)
    
    return clean, noisy, enhanced_demucs, enhanced_spectral_sub #, enhanced_wiener


clean, noisy, enhanced_demucs, enhanced_spectral_sub = getSounds(cfg)


print("PESQ & STOI calculation started")
metrics = {}
metrics['pesq_noisy'] = get_pesq(clean, noisy,16000)
metrics['pesq_enhanced_demucs'] = get_pesq(clean, enhanced_demucs, 16000)
metrics['pesq_enhanced_spectral_sub'] = get_pesq(clean, enhanced_spectral_sub,16000)

metrics['stoi_noisy'] = get_stoi(clean, noisy, 16000)
metrics['stoi_enhanced_demucs'] = get_stoi(clean, enhanced_demucs, 16000)
metrics['stoi_enhanced_spectral_sub'] = get_stoi(clean, enhanced_spectral_sub, 16000)
print("PESQ & STOI calculation done")

def get_model():
    return nemo_asr.models.EncDecCTCModel.from_pretrained(
        model_name='stt_ru_quartznet15x5')

model = get_model()

with open(os.path.join(cfg.results_dir, 'transcriptions.json'),'r') as f:
    transcriptions = json.load(f)
    
def getId(filename):
    return int(filename.split('/')[-1].split('.')[0])


def calc_wer(model, signal_type, cfg):
    """
    signal_type should be one of 'noisy', 'clean', 'enhanced_demucs', 'enhanced_spectral_sub', ...
    """
    
    file_paths = [os.path.join(cfg.results_dir,signal_type,file) for file in os.listdir(os.path.join(cfg.results_dir, signal_type))]
    
    with open(os.path.join(cfg.results_dir, 'transcriptions.json'),'r') as f:
        true_transcriptions = json.load(f)
    
    predicted_transcriptions = model.transcribe(file_paths, batch_size = 16)
    
    true_trans = [""]*len(true_transcriptions)
    pred_trans = [""]*len(predicted_transcriptions)
    
    for i,file in enumerate(file_paths):
        _id = getId(file)
        true_trans[_id] = true_transcriptions[str(_id)]
        pred_trans[_id] = predicted_transcriptions[i]
    
    for i in range(len(pred_trans)):
        pred_trans[i] = normalizeString(pred_trans[i])
        true_trans[i] = normalizeString(true_trans[i])
        
    return get_wer(true_trans, pred_trans)


print("WER calculation started")

metrics['wer_clean'] = calc_wer(model, 'clean',cfg)
metrics['wer_noisy'] = calc_wer(model, 'noisy', cfg)
metrics['wer_enhanced_demucs'] = calc_wer(model, 'enhanced_demucs', cfg)
metrics['wer_enhanced_spectral_sub'] = calc_wer(model, 'enhanced_spectral_sub', cfg)

print("WER calculation done")

metrics['snr'] = cfg.snr
metrics['noise_types'] = cfg.noise_types

with open(os.path.join(cfg.results_dir,'metrics.json'),'w') as f:
    json.dump(metrics, f)

print("Metrics  saved")