import os
import sys
import h5py
import numpy as np
import argparse
from tqdm import tqdm
from subprocess import call
import cPickle as pickle
from python_speech_features import mfcc
import scipy.io.wavfile as wav

parser = argparse.ArgumentParser()
parser.add_argument("split_file", type=str, help="the pickled split file")
parser.add_argument("split", type=str, help="choose the split to use")
parser.add_argument("save_dir", type=str, help="dir to save wavs")
# MFCC options
parser.add_argument("-n", "--nfft", type=int, default=2048, help="the FFT size")
parser.add_argument("-w", "--winlen", type=float, default=0.025, help="the length of the analysis window in seconds")
parser.add_argument("-s", "--winstep", type=float, default=0.01, help="the step between successive windows in seconds")
args = parser.parse_args()


tmp_dir = '/tmp'

done_videos = set()

def extract_wav(all_videos):
    for vid in tqdm(all_videos, ncols=64):
        #vvid = vid.split('/')[-1].split('.')[0]
        vvid, _ = os.path.splitext(vid) # discard extension
        _, vvid = os.path.split(vvid)   # get filename without path
        if vvid in done_videos:
            print 'video %s seen before, ignored.' % vvid

        v_dir = os.path.join(tmp_dir, vvid)
        call(["rm", "-rf", v_dir])
        os.mkdir(v_dir)    # caching directory to store ffmpeg extracted frames

        wav_out = "%s/%s.wav" % (args.save_dir, vvid)
        # Step 1. extract audio
        call(["ffmpeg", "-loglevel", "panic", "-i", vid, "-q:a", "0", "-c:a", "pcm_f32le", "-ac", "1", wav_out])
        if not os.path.exists(wav_out):
            # this video does not have audio channel
            call(["rm", "-rf", v_dir])
            continue

        call(["rm", "-rf", v_dir])
        #call(["rm", "-f", mono_wav])
        done_videos.add(vvid)

if __name__ == "__main__":

    splits = pickle.load(open(args.split_file,'rb'))
    #for split in splits.keys():
    split = args.split
    print splits.keys(), 'using %s' %(split)
    all_videos = splits[split]
    extract_wav(all_videos)
