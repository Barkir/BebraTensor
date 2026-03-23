import gzip
f = gzip.open('t10k-labels-idx1-ubyte.gz','r')

image_size = 28
num_images = 10

import numpy as np

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

import matplotlib.pyplot as plt
for i in range(num_images):
    file = open(f"{i}.png", "w")
    file.write(str(data[i]))
    file.close


