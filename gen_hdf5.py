#-* coding:utf-8 *-
import csv
import os
import h5py
import random
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

args = parser.parse_args()

def get_label_dict():
    f = open('/home/skye/YingYongBao/QA_Exp/vid_url_tag_name_20180804.csv','r')
    csvreader = csv.reader(f)
    cf = open('original_labels.lst','r')
    _dict = {}
    for line in cf.readlines():
        item = line.strip().split(' ')
        _dict[item[0]] = int(item[1])
    cf.close()        

    id_url_label = [line[:3] for line in csvreader] 
    id_url_label = id_url_label[1:]
    _numdict = {}
    
    for item in tqdm(id_url_label):
        vid = item[0]
        labels = item[2]
        labels = labels.split(' ')
        onehot = np.zeros([len(_dict)])
        for l in labels:
            label_digit = _dict[l]
            onehot[label_digit] = 1
        _numdict[vid] = onehot
    return _numdict

def create_onehot_hdf5(ori_hdf5):
    h5_ori = h5py.File(ori_hdf5,'r')
    
    one_hot_hdf5 = ori_hdf5.split('.')[0] + "_onehot.hdf5"
    h5_onehot = h5py.File(one_hot_hdf5,'w')
    
    keys = h5_ori.keys()

    label_dict = get_label_dict()
    valid_keys = label_dict.keys()
    common_keys = list(set(keys) & set(valid_keys))
    random.shuffle(keys)
    
    pbar = tqdm(total = len(common_keys))
    for key in common_keys:
        pbar.update(1)
        #subnames = h5_ori[key]
        #video_data = []
        #for subname in subnames:
        #    video_data.append(h5_ori[key][subname])
        #if len(video_data) != args.frame_num:
        #    continue
        try:
            h5_onehot['data/'+key] = np.array(h5_ori[key]['data'],dtype=np.float32)
            h5_onehot['label/'+key] = label_dict[key]
	    h5_onehot['id/'+key] = np.string_(key)
        except:
            print ("{} already exists".format(key))
            continue
    pbar.close()
    h5_ori.close()
    h5_onehot.close()

if __name__ == "__main__":
    get_label_dict()
    create_onehot_hdf5('all.hdf5')
