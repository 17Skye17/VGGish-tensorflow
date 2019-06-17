# Easy-to-user VGGish Tensorflow
## Introduction

This is an easy-to-use tensorflow version for VGGish.

VGGish is a **VGG-like** audio classification model that pretrained on a large YouTube Dataset.

The official tensorflow version for VGGish model only provides a demo for inferencing a single sample, while this repo provides a **batch** processing version which is much **easier** and **faster** than original repo.

Are you ready to extract your own audio features from original videos? Let's have a try! :)

## Features
- extract audio wav files from video files.
- extract 128-D embedding features from wav files.

## Preparation
### Packages
```shell
pip install \
ffmpeg \
tqdm \
scipy \
h5py \
numpy \
tensorflow \
tensorpack
```

### Downloads
Pretrained VGGish checkpoint: 
https://storage.googleapis.com/audioset/vggish_model.ckpt

Code: 
```shell
git clone https://github.com/17Skye17/VGGish-tensorflow.git
```
## Usage
### 1.Split Video Files
```shell
python split_video_dataset.py \
  --vid_dir=[original video directory] \
  --num_splits=split_num \
  --split_file='split.pkl' 
```
example:
```shell
python split_video_dataset.py \
  --vid_dir='/home/skye/myvideos' \
  --num_splits=4 \
  --split_file='split.pkl' 
```
Then a file named `split.pkl` is generated. This scipt splits videos into 4 parts.

### 2.Extract Audio From Video Files
```shell
python extract_wav.py \
  --split_file='split.pkl' \
  --split='split-0' \
  --save_dir=[audio directory]
```
example:
```shell
python extract_wav.py \
  --split_file='split.pkl' \
  --split='split-0' \
  --save_dir='/home/skye/myaudios'
```
Then the audio files are stored in `myaudios/`.

### 3.Extract VGGish Features
```shell
 cd vggish
 python extract_vggish.py \
    --wav_dir=[audio directory] \
    --save_file=[hdf5 file that stores extracted features] \
    --checkpoint=[path to pretrained VGGish model] \
    --frames=[number of frames per video] \
    --batch_size=[batch_size]
```
example:
```shell
 python extract_vggish.py \
    --wav_dir='/home/skye/myaudios' \
    --save_file='vggish.hdf5' \
    --checkpoint='ckpts/vggish_model.ckpt' \
    --frames=64 \
    --batch_size=128
```

## TODO
Add PCA and Quantization Module

## Ref
https://github.com/tensorflow/models/tree/master/research/audioset

Issues or improvments are welcome, please inform me at skyezx2018@gmail.com if you find any bug.

Update: 06/17/2019

Skye
