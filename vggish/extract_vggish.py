import sys
import os
import h5py
import tensorflow as tf
import multiprocessing
import vggish_input
import vggish_slim
import argparse
import numpy as np
from tensorpack import dataflow
from tqdm import tqdm

#wav_dir = sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument("wav_dir", type=str, help="audio files directory.")
parser.add_argument("save_file", type=str, help="the hdf5 file to save extracted 128-d features.")
parser.add_argument("checkpoint", type=str, help="path to vggish pretrained model.")
parser.add_argument("frames", type=int, help="sample n frames per video.")
parser.add_argument("batch_size", type=int, help="batch size.")

args = parser.parse_args()

videos = os.listdir(args.wav_dir)
videos = [os.path.join(args.wav_dir,video) for video in videos]
#f = open(args.wav_file,'r')
#videos = [v.strip() for v in f.readlines()]

def map(wav):
    
    #try:
        examples_batch = vggish_input.wavfile_to_examples(wav)
        indices = np.linspace(0,len(examples_batch)-1,args.frames).astype(np.int32)
        examples_batch = np.take(examples_batch, indices,axis=0)
        return os.path.basename(wav).split('.')[0],examples_batch
    #except:
    #    wav = '/DATACENTER/3/skye/all_audios/z06727odjnu.wav'
    #    examples_batch = vggish_input.wavfile_to_examples(wav)
    #    indices = np.linspace(0,len(examples_batch)-1,args.frames).astype(np.int32)
    #    examples_batch = np.take(examples_batch, indices,axis=0)
    #    return os.path.basename(wav).split('.')[0],examples_batch

nproc = multiprocessing.cpu_count()/4
nproc = max(20, nproc)
ds = dataflow.DataFromList(videos,shuffle=False)
ds = dataflow.MultiProcessMapDataZMQ(ds, nr_proc=nproc, map_func=map, buffer_size=32, strict=True)

ds.reset_state()

def wrapper():
    for dp in ds.get_data():
        yield tuple(dp)

tf_ds = tf.data.Dataset.from_generator(
        wrapper,
        output_types=(tf.string,tf.float32),
        output_shapes=(tf.TensorShape([]),
                       tf.TensorShape([64,96,64])),
        )

tf_ds = tf_ds.repeat(1).batch(args.batch_size)
tf_ds = tf_ds.prefetch(buffer_size=32)

keys, data = tf_ds.make_one_shot_iterator().get_next()

sess = tf.Session()

layer = vggish_slim.define_vggish_slim(data, training=False)
vggish_var_names = [v.name for v in tf.global_variables()]
vggish_vars = [v for v in tf.global_variables() if v.name in vggish_var_names]

saver = tf.train.Saver(vggish_vars, name='vggish_load_pretrained')
saver.restore(sess,args.checkpoint)


feat_db = h5py.File(args.save_file,'a')
for count in range(1,len(videos)):
    try:
        _id, features = sess.run([keys,layer])
        features = np.reshape(features,[-1,args.frames,128])
        print ("processed {} videos".format(count*args.batch_size))
        count += 1
        for k,o in zip(_id,features):
            feat_db[k] = o
    except:
       print ("this batchsize is broken")
       continue
feat_db.close()
