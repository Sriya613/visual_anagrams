from PIL import Image
import numpy as np
from PIL import ImageOps

import torch

from .view_base import BaseView

class NegateView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        return -im
    
    def apply(self, image):
        return ImageOps.invert(image)   
     
    def __call__(self, image):
        return self.apply(image)

    def inverse_view(self, noise):
        '''
        Negating the variance estimate is "weird" so just don't do it.
            This hack seems to work just fine
        '''
        invert_mask = torch.ones_like(noise)
        invert_mask[:3] = -1
        return noise * invert_mask

    # def make_frame(self, im, t):
    #     im_size = im.size[0]
    #     frame_size = int(im_size * 1.5)

    #     # map t from [0, 1] -> [1, -1]
    #     t = 1 - t
    #     t = t * 2 - 1

    #     # Interpolate from pixels from [0, 1] to [1, 0]
    #     im = np.array(im) / 255.
    #     im = ((2 * im - 1) * t + 1) / 2.
    #     im = Image.fromarray((im * 255.).astype(np.uint8))

    #     # Paste on to canvas
    #     frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
    #     frame.paste(im, ((frame_size - im_size) // 2, (frame_size - im_size) // 2))

    #     return frame

    def make_frame(self, im, t):
        """
        im can be either a tensor (on CUDA) or a PIL Image.
        This version supports both.
        """
        if isinstance(im, torch.Tensor):
            # Tensor format, shape like [C,H,W]
            im_size = im.shape[-1]
            frame_size = int(im_size * 1.5)

            # Ensure t in [-1,1]
            t = 1 - t
            t = t * 2 - 1

            # Interpolate pixel values directly on GPU
            im = ((2 * im - 1) * t + 1) / 2.0
            im = torch.clamp(im, 0, 1)

            # Convert to uint8 tensor for PIL conversion if needed
            im_uint8 = (im * 255).to(torch.uint8)

            # Move to CPU only when converting to PIL for visualization
            im_pil = Image.fromarray(im_uint8.permute(1, 2, 0).cpu().numpy())

        else:
            # PIL image path (for CPU)
            im_size = im.size[0]
            frame_size = int(im_size * 1.5)

            t = 1 - t
            t = t * 2 - 1

            im = np.array(im) / 255.
            im = ((2 * im - 1) * t + 1) / 2.
            im = Image.fromarray((im * 255.).astype(np.uint8))
            im_pil = im

        # Paste on to frame canvas
        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        frame.paste(im_pil, ((frame_size - im_pil.size[0]) // 2, (frame_size - im_pil.size[1]) // 2))

        return frame