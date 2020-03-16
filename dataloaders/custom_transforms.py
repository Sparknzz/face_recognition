import torch
import random
import numpy as np
import cv2


def do_short_resize(image, size):
        
    h, w, _ = image.shape

    if h > w:
        new_h, new_w = size * h // w, size
    else:
        new_h, new_w = size, size * w // h
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def do_flip_lr(image):
    
    image = cv2.flip(image, 1)

    return image


def brightness_aug(self, src, x):
    alpha = 1.0 + random.uniform(-x, x)
    src *= alpha
    return src

def contrast_aug(self, src, x):
    alpha = 1.0 + random.uniform(-x, x)
    coef = nd.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
    src *= alpha
    src += gray
    return src

def saturation_aug(self, src, x):
    alpha = 1.0 + random.uniform(-x, x)
    coef = nd.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray = nd.sum(gray, axis=2, keepdims=True)
    gray *= (1.0 - alpha)
    src *= alpha
    src += gray
    return src
