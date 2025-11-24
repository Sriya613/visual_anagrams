from PIL import Image

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch
from .view_base import BaseView


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def apply(self, image):
        return image.rotate(90)

    def __call__(self, image):
        return self.apply(image)

    def view(self, im):
        # TODO: Is nearest-exact better?
        return TF.rotate(im, -90, interpolation=InterpolationMode.NEAREST)

    def inverse_view(self, noise):
        return TF.rotate(noise, 90, interpolation=InterpolationMode.NEAREST)

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * -90

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame


class Rotate90CCWView(BaseView):
    def __init__(self):
        pass

    def apply(self, image):
        return image.rotate(270)

    def __call__(self, image):
        return self.apply(image)
    
    def view(self, im):
        # TODO: Is nearest-exact better?
        return TF.rotate(im, 90, interpolation=InterpolationMode.NEAREST)

    def inverse_view(self, noise):
        return TF.rotate(noise, -90, interpolation=InterpolationMode.NEAREST)

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * 90

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame

#chatgpt fix for the attribute error we got

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class Rotate180View(BaseView):
    def __init__(self):
        pass

    # Remove .apply that uses PIL — not needed for tensor flow
    def __call__(self, image):
        return self.view(image)

    def view(self, im):
        # Works on tensors (N,C,H,W) or (C,H,W)
        if isinstance(im, torch.Tensor):
            return TF.rotate(im, 180, interpolation=InterpolationMode.NEAREST)
        else:
            return im.rotate(180)

    def inverse_view(self, noise):
        if isinstance(noise, torch.Tensor):
            return TF.rotate(noise, -180, interpolation=InterpolationMode.NEAREST)
        else:
            return noise.rotate(-180)

    def make_frame(self, im, t):
        im_size = im.size[0]
        # print("roatte ")
        frame_size = int(im_size * 1.5)
        theta = t * 180

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame
