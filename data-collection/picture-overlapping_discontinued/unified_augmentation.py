
'''
This script holds data augmentation functions for creating variants of the
Sources that were to be used in the picture overlapping dataset.

DISCONTINUED
'''

cfg = {
    'items_path': './data/',
    'background_path': './backgrounds/',
    'out_path': './example/',

    'classnames_goofy': ['backp', 'survivor', 'phone', 'fire', 'drill', 'helm'],

    'ouput_num': 600,   # total number of output pictures
    'items_per_im': 5,  # number of items per image
    'pic_size': 320,    # output picture size

    'rotation': 30,     # rotation range
    'resize': [],       # normal dist
    'mirror': [h, v],   # bool
    'brightness': [],   # normal dist
    'recolor': [],      # red / yellow / white light
    }



def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def unified_augment(img, cfg):
    img_rgb = img[:,:,:3]
    img_alpha = img[:,:,3]



X = 0
bg = os.listdir(cfg['background_path'])
img = cv2.imread(cfg['background_path'] + bg[X])

cv2.imshow('original', img)
cv2.imshow('aug', img_out)
