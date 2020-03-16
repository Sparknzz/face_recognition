import os
import pandas as pd

# webface 250*250
face_dir = '/home/data/CASIA/CASIA-WebFace'
i = -1

all_img = []
all_label = []

for root, _, files in os.walk(face_dir):
    img_path = [os.path.join(root, file) for file in files]
    label = [i for _ in files]

    all_img += img_path
    all_label += label
    
    i+=1

print('Number of images : {:d}'.format(len(all_img)))
print('Number of people : {:d}'.format(i+1))

df = pd.DataFrame({'image':all_img, 'label':all_label})
df = df.sample(frac=1)

df.to_csv('./CASIA.csv', index=False)
