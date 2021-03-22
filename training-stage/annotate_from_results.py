import cv2
import colorsys

out_dir = '/home/pablo/YOLOv4/infer_output/'
text_file_path =  '/home/pablo/YOLOv4/darknet/results_pablo.txt'
idx = 0

thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5

with open(text_file_path, 'r') as file:
    for line in file.readlines():

        if '/home/pablo/' in line:
            if idx != 0:
                cv2.imwrite(f'{out_dir + str(idx)}.jpg', pic)
                print(f'Saved at {out_dir + str(idx)}.jpg')
            picpath = line.split(':')[0]
            pic = cv2.imread(picpath)
            picsize_original = pic.shape[0]
            pic = cv2.resize(pic, (680,680))
            idx += 1

        if '%' in line:
            title = line.split('%')[0] + '%'
            hue = ord(title[0]) - (100.5 - ord(title[0])) * 15
            color = tuple(255 * i  for i in colorsys.hsv_to_rgb(hue/360.0, 1, 1))

            scalefactor = 680 / picsize_original
            x1, y1, w, h = [int(abs(int(i))*scalefactor) for i in line[:-2].replace('-','').split() if i.isdigit()]
            pic = cv2.rectangle(pic, (x1, y1), (x1 + w, y1 + h), color, thickness)
            pic = cv2.putText(pic, title, (x1, y1), font, fontScale, color, thickness, cv2.LINE_AA)
