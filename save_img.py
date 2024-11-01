import torch
import torchvision.utils as vutils

## a의 tensorsize : (64,3,256,128)
a = (a - a.min()) / (a.max() - a.min()) #정규화
for i in range(a.size(0)):
    vutils.save_image(a[i], f'output_image_{i}.png', normalize=False)
