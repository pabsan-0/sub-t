import numpy as np
import cv2
import time
import random
import os

''' 
This script will place some items on top of a background picture &
generate a .txt with YOLO bounding box annotations.

TODO: check that all categories are included
TODO: implement augmentations: SIZE; MIRRORING; ROTATION; BRIGHTNESS; LIGHT COLOR
'''


cfg = {
    'items_path': './data/',
    'background_path': './backgrounds/',
    'out_path': './example/',
    'ouput_num': 600,
    'items_per_im': 5,
    'pic_size': 320,
    'rotation': 30,
    }


def randomint(max, min=0, step=1):
    """Defaults some values for the inbuilt function."""
    return random.randrange(min, max, step)


def centercrop(pic, size):
    """ Center-crops a picture in a square shape of given size. """
    x = pic.shape[1]/2 - size/2
    y = pic.shape[0]/2 - size/2
    return pic[int(y):int(y+size), int(x):int(x+size)]


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by pos
    and blend using alpha_mask. Alpha mask must contain values within the
    range [0, 1] and be the same size as img_overlay.

    Note that this function already will modify the input image. The return
    variable was added mainly for debugging.
    """
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])
    return img, [x1, y1, x2, y2]


def choosecat(str):
    """ Gets a string and tries to match its category with an int. """
    ids = ['backp', 'survivor', 'phone', 'fire', 'drill', 'helm']
    for i, id in enumerate(ids):
        if id in str:
            return i
    return None


def get_random_crop(image, crop_height, crop_width):

    # determine picture limits & randomly choose where to crop
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    # actually crop the thing
    crop = image[y: y + crop_height, x: x + crop_width]

    return crop



if __name__ == '__main__':
    # Loop as many times as pictures we want
    for j in range(cfg['ouput_num']):

        # Choose background + crop to square shape + add alpha
        bg = random.choice(os.listdir(cfg['background_path']))
        bg = cv2.imread(cfg['background_path'] + bg)
        bg = get_random_crop(bg, cfg['pic_size'], cfg['pic_size'])
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2RGBA)

        # Randomly pick a few items to paste on picture
        candidates = os.listdir(cfg['items_path'])
        fg = []
        for i in range(cfg['items_per_im']):
            fg.append(random.choice(candidates))

        # Init empty annotations for this picture
        pic_annotation = []

        # Add items one by one
        for i, item_path in enumerate(fg):

            # Retrieve the item pic and de-scale it
            item = cv2.imread(cfg['items_path'] + item_path, -1)
            item = cv2.pyrDown(cv2.pyrDown(item))

            # Get height & width of item & randomly place it
            h, w  = item.shape[:2]
            x = randomint(cfg['pic_size']) - w//2
            y = randomint(cfg['pic_size']) - h//2

            # Place this item on top of the picture & retrieve patch coordinates
            pic, bbox = overlay_image_alpha(
                            bg,
                            item[:, :, :],
                            (x, y),
                            item[:, :, 3] / 255.0)

            # Unpack coordinates & craft YOLO annotations
            x1, y1 = bbox[:2]
            x2, y2 = bbox[2:]
            w , h  = x2-x1, y2-y1
            imsize = cfg['pic_size']
            cat = choosecat(item_path)
            pic_annotation.append(
                f'{cat} {x1/imsize} {y1/imsize} {w/imsize} {h/imsize}')

        # Write annotations to file & export results
        with open(cfg['out_path'] + str(j) + '.txt', 'w') as file:
            for item_info in pic_annotation:
                file.write(item_info + '\n')
        cv2.imwrite(cfg['out_path'] + str(j) + '.jpeg', pic)
