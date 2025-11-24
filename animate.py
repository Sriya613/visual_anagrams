from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat
import imageio

import torch
import torchvision.transforms.functional as TF

from visual_anagrams.utils import get_courier_font_path


# def draw_text(image, text, fill=(0,0,0), frame_size=384, im_size=256):
#     """
#     Draw `text` centered below the visible image area inside `image`.
#     Automatically detects the non-background bbox (works for white / near-white bg).
#     """
#     img = image.copy()
#     draw = ImageDraw.Draw(img)

#     # Font scaling
#     font_path = get_courier_font_path()
#     base_font_size = 16
#     font_size = max(10, int(base_font_size * frame_size / 384))
#     font = ImageFont.truetype(font_path, font_size)

#     # --- Detect non-background bounding box ---
#     # Convert to grayscale and threshold to find non-white content.
#     # Tweak threshold if your backgrounds are not pure white.
#     gray = img.convert("L")
#     # Compute a dynamic threshold: if image is mostly white, keep 250; else 240
#     stat = ImageStat.Stat(gray)
#     mean_lum = stat.mean[0]
#     thresh = 250 if mean_lum > 245 else 240

#     # Create mask of "non-background" pixels
#     mask = gray.point(lambda p: 0 if p > thresh else 255).convert("1")
#     bbox = mask.getbbox()  # (left, upper, right, lower) or None

#     # If bbox is None (no content detected), fallback to central square
#     if bbox is None:
#         # assume the image content is a centered square of size im_size
#         left = (frame_size - im_size) // 2
#         right = left + im_size
#         top = (frame_size - im_size) // 2
#         bottom = top + im_size
#     else:
#         left, top, right, bottom = bbox

#     # --- Measure text size precisely ---
#     text_bbox = draw.textbbox((0, 0), text, font=font)
#     text_w = text_bbox[2] - text_bbox[0]
#     text_h = text_bbox[3] - text_bbox[1]

#     # Horizontal: center under the detected content bbox
#     text_x = left + ( (right - left) - text_w ) // 2
#     # Clamp so it stays inside the frame
#     text_x = max(0, min(text_x, frame_size - text_w))

#     # Vertical: place in the space below the content bbox, centered inside that band.
#     # If there's little space under content, nudge it a bit down.
#     bottom_space = frame_size - bottom
#     if bottom_space <= 0:
#         # no bottom band: place text just below bbox clipped inside frame
#         text_y = min(frame_size - text_h, bottom + 4)
#     else:
#         text_y = bottom + (bottom_space - text_h) // 2

#     # Final safety clamp
#     text_y = max(0, min(text_y, frame_size - text_h))

#     # Draw the text
#     draw.text((text_x, text_y), text, font=font, fill=fill)

#     return img


# def draw_text(image, text, fill=(0,0,0), frame_size=384, im_size=256):--orgiinal animate code
#     image = image.copy()

#     print(image,text,frame_size,im_size)
#     # Font info. Use 16pt for 384 pixel image, and scale up accordingly
#     font_path = get_courier_font_path()
#     font_size = 16
#     font_size = int(font_size * frame_size / 384)

#     # Make PIL objects
#     draw = ImageDraw.Draw(image)
#     font = ImageFont.truetype(font_path, font_size)
    
#     # Center text horizontally, and vertically between
#     # illusion bottom and frame bottom
#     text_position = (0, 0)
#     bbox = draw.textbbox(text_position, text, font=font, align='center')
#     text_width = bbox[2] - bbox[0]
#     text_height = bbox[3] - bbox[1]
#     text_left = (frame_size - text_width) // 2


#     # text_top = int(3/4 * frame_size + 1/4 * im_size - 1/2 * text_height)
#     # Since CombinedView puts the illusion in top 75% and text area in bottom 25%
#     text_top = int(0.75 * frame_size + (0.25 * frame_size - text_height) / 2)

#     # margin = (frame_size - im_size) // 2
#     # text_top = margin + im_size + 10  # 10px below the image area

    
#     text_position = (text_left, text_top)

#     # Draw text on image
#     draw.text(text_position, text, font=font, fill=fill, align='center')
#     return image
def draw_text(image, text, fill=(0,0,0), im_size=None):
    # image: PIL Image
    image = image.copy()
    w, h = image.size

    # Determine bottom area
    bottom_top = int(0.75 * h)
    bottom_h = h - bottom_top

    # Font scaling: base font 16 at width 384
    font_path = get_courier_font_path()
    base_font = 16
    font_size = max(10, int(base_font * w / 384))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)

    # Wrap text to fit width (use ~90% of frame width)
    max_text_width = int(w * 0.9)
    words = text.split()
    lines = []
    if not words:
        lines = [""]
    else:
        cur = words[0]
        for word in words[1:]:
            test = cur + " " + word
            bbox = draw.textbbox((0,0), test, font=font)
            if bbox[2] - bbox[0] <= max_text_width:
                cur = test
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)

    # Compute total text block height
    line_heights = []
    line_widths = []
    for ln in lines:
        bbox = draw.textbbox((0,0), ln, font=font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        line_widths.append(lw)
        line_heights.append(lh)
    spacing = max(2, font_size // 6)
    text_block_h = sum(line_heights) + spacing * (len(lines) - 1)

    # Vertical start: center the block inside the bottom 25%
    y_start = bottom_top + max(0, (bottom_h - text_block_h) // 2)

    # Draw each line centered horizontally
    y = y_start
    for ln, lw, lh in zip(lines, line_widths, line_heights):
        x = (w - lw) // 2
        draw.text((x, y), ln, font=font, fill=fill, align='center')
        y += lh + spacing

    return image

def easeInOutQuint(x):
    # From Matthew Tancik: 
    # https://github.com/tancik/Illusion-Diffusion/blob/main/IllusionDiffusion.ipynb
    if x < 0.5:
        return 4 * x**3
    else:
        return 1 - (-2 * x + 2)**3 / 2


def animate_two_view(
        im,
        view,
        prompt_1,
        prompt_2,
        save_video_path,
        hold_duration=120,
        text_fade_duration=10,
        transition_duration=60,
        im_size=256,
        frame_size=384,
        boomerang=True,
):
    '''
    Animate the transition between an image and the view of an image

    im (PIL Image):
        Image to animate

    view (view object):
        The view to transform the image by. Importantly, 
        should implement `make_frame`.

    prompt_1, prompt_2 (string):
        Prompt for the identity view and the transformed view, to be
        displayed under the images

    save_video_path (string):
        Path to the location to save the video

    hold_duration (int):
        Number of frames (@ 30 FPS) to pause the video on the
        complete image

    text_fade_duration (int):
        Number of frames (@ 30 FPS) for the text to fade in or out

    transition_duration (int):
        Number of frames (@ 30 FPS) to animate the transformations

    im_size (int):
        Size of the image to animate

    frame_size (int):
        Size of the final video

    boomerang (bool):
        If true, boomerang the clip by showing image, then transformed
        image, and finally the original image.
        If false, only show image then transformed image.
    '''

    # Make list of frames
    frames = []

    # Make frames for two views 
    frame_1 = view.make_frame(im, 0.0)
    frame_2 = view.make_frame(im, 1.0)

    print(view,frame_1,frame_2,"this is view and frame")
    # Display frame 1 with text
    frame_1_text = draw_text(frame_1,
                             prompt_1,
                             im_size=im_size)
    frames += [frame_1_text] * (hold_duration // 2)

    # Fade out text 1
    for t in np.linspace(0,1,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
    frame = draw_text(frame_1,
              prompt_1,
              fill=fill,
              im_size=im_size)
    frames.append(frame)

    # Transition view 1 -> view 2
    print('Making frames...')
    for t in tqdm(np.linspace(0,1,transition_duration)):
        t_ease = easeInOutQuint(t)
        frames.append(view.make_frame(im, t_ease))

    # Fade in text 2
    for t in np.linspace(1,0,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
    frame = draw_text(frame_2,
              prompt_2,
              fill=fill,
              im_size=im_size)
    frames.append(frame)

    # Display frame 2 with text
    frame_2_text = draw_text(frame_2,
                             prompt_2,
                             im_size=im_size)
    frames += [frame_2_text] * (hold_duration // 2)

    # "Boomerang" the clip, so we get back to view 1
    if boomerang:
        frames = frames + frames[::-1]

    # Move last bit of clip to front
    frames = frames[-hold_duration//2:] + frames[:-hold_duration//2]

    # Convert PIL images to numpy arrays
    image_array = [imageio.core.asarray(frame) for frame in frames]

    # Save as video
    print('Making video...')
    imageio.mimsave(save_video_path, image_array, fps=30)




####################################
### ANIMATION FOR MOTION HYBRIDS ###
####################################

def easedLinear(t):
    '''
    Ramps up into a linear increase
    '''
    if t < np.sqrt(1/3):
        return t**3
    else:
        return t - np.sqrt(1/3) + np.power(1/3, 3/2)

def animate_two_view_motion_blur(
        im,
        view,
        prompt_1,
        prompt_2,
        save_video_path,
        hold_duration=120,
        text_fade_duration=10,
        transition_duration=60,
        im_size=256,
        frame_size=384,
        boomerang=True,
        text_top=None,
):
    '''
    Animate the transition between an image and the view of an image

    im (PIL Image):
        Image to animate

    view (view object):
        The view to transform the image by. Importantly, 
        should implement `make_frame`.

    prompt_1, prompt_2 (string):
        Prompt for the identity view and the transformed view, to be
        displayed under the images

    save_video_path (string):
        Path to the location to save the video

    hold_duration (int):
        Number of frames (@ 30 FPS) to pause the video on the
        complete image

    text_fade_duration (int):
        Number of frames (@ 30 FPS) for the text to fade in or out

    transition_duration (int):
        Number of frames (@ 30 FPS) to animate the transformations

    im_size (int):
        Size of the image to animate

    frame_size (int):
        Size of the final video

    boomerang (bool):
        If true, boomerang the clip by showing image, then transformed
        image, and finally the original image.
        If false, only show image then transformed image.

    text_top (int):
        Adjust vertical location of text for second view. For hybrid images
    '''

    # Make list of frames
    frames = []

    # Make frames for two views 
    frame_1 = view.make_frame(im, 0.0)
    frame_2 = view.make_frame(im, 1.0)

    # Display frame 1 with text
    frame_1_text = draw_text(frame_1,
                             prompt_1,
                             im_size=im_size)
    frames += [frame_1_text] * (hold_duration // 2)

    # Fade out text 1
    for t in np.linspace(0,1,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
    frame = draw_text(frame_1,
              prompt_1,
              fill=fill,
              im_size=im_size)
    frames.append(frame)

    # Transition view 1 -> view 2
    # 1. Make buffer of frames to blur together
    print('Making frames...')
    blur_buffer = []    
    for t in tqdm(np.linspace(0,2,transition_duration)):
        if t < 1.5:
            t_ease = easedLinear(t)
            blur_buffer.append(view.make_frame(im, t_ease))

    # 2. Make blurred frames
    n = int(transition_duration / 20.)  # T / vel
    blurred_frames = []
    for i in tqdm(range(0, len(blur_buffer) - n, 10)):
        if i <= n * 3:
            window = 1
        else:
            window = min(int((i - 3 * n) / 10.), n)

        to_blur = blur_buffer[i:i+window]
        to_blur = [TF.to_tensor(im) for im in to_blur]
        im_blurred = torch.mean(torch.stack(to_blur), dim=0)
        blurred_frames.append(TF.to_pil_image(im_blurred))
    frames.extend(blurred_frames)

    # Fade in text 2
    frame_2 = frames[-1]
    for t in np.linspace(1,0,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
    frame = draw_text(frame_2,
              prompt_2,
              fill=fill,
              im_size=text_top if text_top is not None else im_size)
    frames.append(frame)

    # Display frame 2 with text
    frame_2_text = draw_text(frame_2,
                             prompt_2,
                             im_size=text_top if text_top is not None else im_size)
    frames += [frame_2_text] * (hold_duration // 2)

    # "Boomerang" the clip, so we get back to view 1
    if boomerang:
        frames = frames + frames[::-1]

    # Move last bit of clip to front
    frames = frames[-hold_duration//2:] + frames[:-hold_duration//2]

    # Convert PIL images to numpy arrays
    image_array = [imageio.core.asarray(frame) for frame in frames]

    # Save as video
    print('Making video...')
    imageio.mimsave(save_video_path, image_array, fps=30)