import numpy as np
import cv2
import bcolz
import pickle
import mxnet as mx

def load_bin(bin_path, out_path, image_size = [112, 112]):

    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')

    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype = np.float32, rootdir = out_path, mode = 'w')
    
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, dsize=(image_size[0], image_size[1]),)
        data[i, ...] = img.reshape(3, image_size[0], image_size[1])
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)

    print(data.shape)
    np.save(str(out_path) + '_list', np.array(issame_list))
    
    return data, issame_list


if __name__=='__main__':
    eval_bin = '/root/face_recognition/data/eval/lfw.bin'
    out_path = '/root/face_recognition/data/eval/lfw'

    load_bin(eval_bin, out_path)