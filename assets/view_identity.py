# from .view_base import BaseView

# class IdentityView(BaseView):
#     def __init__(self):
#         pass

#     def view(self, im):
#         return im

#     def inverse_view(self, noise):
#         return noise

#     def __call__(self, im):
#         return self.view(im)

from .view_base import BaseView
from PIL import Image
import torch
# If you want to support PIL<->tensor conversions you may import TF_1 used elsewhere:
# import torchvision.transforms.functional as TF_1

class IdentityView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        return im

    def inverse_view(self, noise):
        return noise

    def __call__(self, im):
        return self.view(im)

    def make_frame(self, im, t):
        """
        Return a frame for parameter t in [0,1].
        For the identity view this is simply the original image.
        Handles both PIL.Image and torch.Tensor inputs.
        """
        # If it's a tensor, just return it unchanged
        if isinstance(im, torch.Tensor):
            return im

        # Otherwise assume PIL.Image (most of your make_frame implementations return PIL)
        # For identity there is nothing to interpolate — just return the input image.
        # If you want a smooth crossfade in the future, uncomment the blend code below.
        return im

        # --- optional crossfade example (unnecessary for identity since view(im) == im) ---
        # transformed = self.view(im)
        # if t <= 0:
        #     return im
        # elif t >= 1:
        #     return transformed
        # else:
        #     from PIL import Image
        #     return Image.blend(im, transformed, alpha=t)
