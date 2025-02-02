import os
import numpy as np
from PIL import Image
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.models.modnet import MODNet

import gradio as gr

ref_size = 512

im_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

pretrained_ckpt = 'pretrained/modnet_photographic_portrait_matting.ckpt'
if not os.path.exists(pretrained_ckpt):
    url = "https://drive.google.com/uc?id=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz"
    gdown.download(url, pretrained_ckpt)

modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)  # add .cuda() for gpu
modnet.load_state_dict(torch.load(pretrained_ckpt, map_location='cpu'))  # remove map_location if gpu
modnet.eval()


def inference(im):
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]
    im_np = im
    im = Image.fromarray(im)
    im = im_transform(im)
    im = im[None, :, :, :]

    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im, inference=False)  # add .cuda() to im for gpu

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    # matte_mask = np.repeat(matte[:, :, None], 3, axis=2)
    # foreground = im_np * matte_mask + (1 - matte_mask) * 255
    foreground = np.concatenate((im_np, matte[:, :, None] * 255), axis=2)
    foreground = Image.fromarray(foreground.astype('uint8'))
    return foreground, matte


inputs = gr.inputs.Image(label="Portrait Image")
outputs = [gr.outputs.Image(label="Foreground Image"), gr.outputs.Image(label="Alpha Image")]

title = "MODNet: Is a Green Screen Really Necessary for Real-Time Portrait Matting?"
description = "This is a portrait image matting demo based on MODNet. Try it by uploading your image or clicking one of " \
              "the examples. The uploaded image will not be saved or stored anywhere. Please check out the paper and " \
              "repository for more details (linked at the bottom)."
examples = [
    ["example_images/1.jpg"],
    ["example_images/2.jpg"]
]

article = "http://raw.githubusercontent.com/aliabd/modnet/master/README.md"

gr.Interface(inference, inputs, outputs, title=title, description=description, examples=examples, article=article,
             allow_flagging=False).launch(share=True)
