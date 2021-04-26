from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
from jiwer import wer

def get_pesq(ref_sig, out_sig, sr):
    pesq_val = 0
    for i in tqdm(range(len(ref_sig))):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val/len(ref_sig)


def get_stoi(ref_sig, out_sig, sr):
    stoi_val = 0
    for i in tqdm(range(len(ref_sig))):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val/len(out_sig)

def get_wer(true_transcriptons, predicted_transcriptions):
    return wer(true_transcriptons, predicted_transcriptions)
