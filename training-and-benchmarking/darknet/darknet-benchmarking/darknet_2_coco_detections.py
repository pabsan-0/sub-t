from sys import argv
import os
import json

''' Creates a json file in COCO format for the predictions of darknet.

ARGS:
    ground-truth file in COCO json
    detection file in darknet default (see preview below)
    name of the output detection file in COCO json (will overwrite)

Expected predictions on input file are of this shape...:

/home/pablo/YOLOv4/PPU-6/test/25_exyn_bag15_rgb_frame1500305.png: Predicted in 134.026000 milli-seconds.
drill: 99%	(left_x:  406   top_y:  344   width:   42   height:   53)
survivor: 100%	(left_x:  559   top_y:  303   width:  119   height:  268)
'''


def get_path2id_dict(ground_truth_json):
    ''' Reads ground truth json and links image paths to image ids '''
    path2id = {}
    with open(ground_truth_json) as gt:
        for image in json.load(gt)['images']:
            name = image['file_name']
            id = image['id']
            path2id[name] = id
    return path2id


if __name__ == '__main__':
    # get ground truth file and detection file from input arguments
    gt_file       = argv[1]
    dets_file_in  = argv[2]
    dets_file_out = argv[3]

    # build a dictionary that links filenames to image ids defined in the GT json
    path2id = get_path2id_dict(gt_file)

    # build a dictionary to link each item to its coco ID
    names2ids = {
        'backpack': 1,
        'helmet': 2,
        'drill': 3,
        'extinguisher': 4,
        'survivor': 5,
        'rope': 6,
        }

    # define container for holding the output
    massive_output_list = []

    # read the file holding the darknet detections and scan it
    with open(dets_file_in, 'r') as file:
        for line in file.readlines():

            if '/home/' in line:
                # fetch the image id from the image path that is read
                current_image_path = line.split(':')[0].split('/')[-1]
                current_image_id = path2id[current_image_path]

            if '%' in line:
                # extract category and score by using % as separator
                line_modified = line.replace(':', '%')
                category_id = names2ids[line_modified.split('%')[0]]
                score    = int(line_modified.split('%')[1])/100

                # extract bounding box numbers by doing some complex juggling
                x1, y1, w, h = [int(i) for i in line[:-2].replace('-','').split() if i.isdigit()]

                massive_output_list.append(
                    {
                    'image_id':     current_image_id,
                    'category_id':  category_id,
                    'bbox':         [x1, y1, w, h],
                    'score':        score,
                    }
                )

    # write to output json file AS STRING else it breaks later!
    with open(dets_file_out, 'w') as outfile:
        string_content = json.dumps(massive_output_list)
        outfile.write(string_content)
