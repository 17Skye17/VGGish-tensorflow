import os

base_dir = '/DATACENTER/3/skye/all_audios'
videos = os.listdir(base_dir)
split_len = len(videos)/4

for i in range(5):
    f = open('audio_'+str(i)+'.lst','w')
    for v in videos[split_len*i:split_len*(i+1)]:
        f.write(os.path.join(base_dir,v)+'\n')
    f.close()
    
