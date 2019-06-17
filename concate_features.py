import sys
import h5py
from tqdm import tqdm
import numpy as np

hdf5_file = sys.argv[1]
vgg_file = sys.argv[2]
save_file = sys.argv[3]
inceptionv3_file = sys.argv[4]

frame_num = 64
h5 = h5py.File(hdf5_file,'r')
vggh5 = h5py.File(vgg_file,'r')
h5_concate = h5py.File(save_file,'w')
incept_h5 = h5py.File(inceptionv3_file,'r')

vgg_keys = set(vggh5.keys())
keys = set(h5.keys())
incepv3_keys = incept_h5.keys()
incepv3_keys = list(set(keys) & set(incepv3_keys))

for key in tqdm(incepv3_keys):
    subnames = incept_h5[key].keys()
    vid_data =[]
    for subname in subnames:
        vid_data.append(incept_h5[key][subname])
    vid_data = np.array(vid_data)
    if key in vgg_keys:
        vgg_data = vggh5[key]
    else:
        vgg_data = np.zeros([frame_num,128])
    
    if key in keys:
        user_data = h5[key]['data'][:,2048+256:2404]
    else:
        user_data = np.zeros([frame_num,100])
    concate_data = np.concatenate((vid_data, vgg_data, user_data),axis=1)
    h5_concate['data/'+key] = concate_data
    h5_concate['id/'+key] = np.array(h5[key]['id'])
    h5_concate['label/'+key] = np.array(h5[key]['label'])

h5.close()
h5_concate.close()
vggh5.close()
incept_h5.close()
#for key in tqdm(keys):
#    all_data = np.array(h5[key]['data'])
#    vid_data = all_data[:,:2048]
#    audio_data = all_data[:,2048:2048+256]
#    user_data = all_data[:,2048+256:2404]
#    if key in vgg_keys:
#        #subnames = vggh5[key].keys()
#        #vgg_data = []
#        #for subname in subnames:
#        #    vgg_data.append(np.array(vggh5[key][subname]))
#        #vgg_data = np.array(vgg_data)
#        vgg_data = vggh5[key]
#    else:
#        vgg_data = np.zeros([frame_num,128])
#    concate_data = np.concatenate((vid_data,vgg_data,user_data),axis=1)
#    h5_concate['data/'+key] = concate_data
#    h5_concate['id/'+key] = np.array(h5[key]['id'])
#    h5_concate['label/'+key] = np.array(h5[key]['label'])
#h5.close()
#h5_concate.close()
#vggh5.close()
