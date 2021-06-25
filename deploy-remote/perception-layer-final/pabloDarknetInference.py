################################################################################
#    main-multiprocessing.py / pabloDarknetInference                           #
#    Pablo Santana - CNNs for object detection in subterranean environments    #
################################################################################
# p0 (main)         p1                   p2                      p5
# ┌────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌──────────────┐
# │ImageCapture│ q1 │Estimate Camera│ q3 │Kalman-Filter    │q4  │Draw Worldmap │
# │&& Undistort├┬──>│Pose From ArUco├───>│Camera Pose      ├──┬>│with cam FoV &│
# │            ││   │markers        │    │                 │  │ │item positions│
# └────────────┘│   └───────────────┘    └─────────────────┘  │ └──────────────┘
#               │   p3                   p4                   │
#               │   ┌────────────────┐   ┌─────────────────┐  │
#               │q2 │Object detection│q5 │Estimate item    │q6│
#               └──>│with Darknet    ├──>│position w.r to  ├──┘
#                   │(CNN)           │   │the camera       │
#                   └────────────────┘   └─────────────────┘
#                        └─YOU'RE HERE

import cv2
import os
import sys

'''
Detection shape:

[('bottle',
  '83.45',
  (181.35324096679688,
   214.51708984375,
   41.65458679199219,
   144.4390869140625))]
'''

def mindYourDarknet(func):
    ''' Decorator handling the proper importing of darknet.
    '''
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except ModuleNotFoundError:
            # If path hasn't been changed yet (darknet invisible to python)
            sys.path.append(os.getcwd())
            os.chdir('/home/pablo/YOLOv4/darknet')
            sys.path.insert(1, '/home/pablo/YOLOv4/darknet')
            return func(*args, **kwargs)

        except OSError:
            # Deals with the LD library path for CUDNN, run once per terminal execution
            print('''>>> Catched CUDNN error! Run this bash command and try again:\n
            \rexport LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib
            ''')
            quit()

    return inner_function


@mindYourDarknet
def networkBoot():
    ''' Boot a network model
    '''
    import darknet

    # load a nnet params
    config_file = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.cfg'
    data_file = '/home/pablo/YOLOv4/darknet/cfg/coco.data'
    weights = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.weights'

    # initialize object detection neural network
    network, class_names, colors = darknet.load_network(config_file, data_file, weights, batch_size=1)
    return network, class_names, colors



@mindYourDarknet
def imageDetection(image, network, class_names, class_colors, thresh=0.1):
    '''
    Modified version of the function from darknet_images.py that accepts
    a live frame instead of a file path for loading an image, conveting it to
    C format and performing Inference.
    '''
    import darknet

    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    # image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections



''' DEPRECATED BUT STORED TOWARDS CLEANER IMPORT
def prepareImportDarknet():
    # VERY NASTY way of importing the darknet bindings without installing extra packages
    sys.path.append(os.getcwd())
    os.chdir('/home/pablo/YOLOv4/darknet')
    sys.path.insert(1, '/home/pablo/YOLOv4/darknet')
    try:
        import darknet

    except OSError:
        # deals with the LD library path for CUDNN, run once per terminal execution
        print(""">>> Catched CUDNN error! Run this bash command and try again:\n
        \rexport LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib
        """)
        quit()
'''
