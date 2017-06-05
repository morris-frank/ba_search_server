#!/usr/bin/env python3
from tqdm import tqdm
import scipy.misc
from glob import glob
import os
import ba.utils


p  = '/net/hci-storage01/groupfolders/compvis/mfrank/arthistoric_images/imageFiles_8/'

d = {}
for ip in tqdm(glob(p + '*png')):
    bn = os.path.splitext(os.path.basename(ip))[0]
    im = scipy.misc.imread(ip)
    shape = im.shape[:2]
    d[bn] = shape
ba.utils.save(p[:-1] + '_sizes.yaml', d)
