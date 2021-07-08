################################################################################
#    main-multiprocessing.py / pabloWorldMap                                   #
#    Pablo Santana - CNNs for object detection in subterranean environments    #
################################################################################
# p0 (main)         p1                   p2                      p5
# ┌────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌──────────────┐
# │ImageCapture│ q1 │Estimate Camera│ q3 │Kalman-Filter    │q4  │Draw Worldmap │
# │&& Undistort├┬──>│Pose From ArUco├───>│Camera Pose      ├──┬>│with cam FoV &│
# │            ││   │markers        │    │                 │  │ │item positions│
# └────────────┘│   └───────────────┘    └─────────────────┘  │ └──────────────┘
#               │   p3                   p4                   │    └─YOU'RE HERE
#               │   ┌────────────────┐   ┌─────────────────┐  │
#               │q2 │Object detection│q5 │Estimate item    │q6│
#               └──>│with Darknet    ├──>│position w.r to  ├──┘
#                   │(CNN)           │   │the camera       │
#                   └────────────────┘   └─────────────────┘
#

import numpy as np
from numpy.linalg import inv
import cv2


def rotationMatrix(theta, unit='rad'):
    if unit != 'rad':
        theta = np.deg2rad(theta)
    return np.array([[ np.cos(theta), -np.sin(theta)],
                     [ np.sin(theta), np.cos(theta)]])


def translationMatrix(dx, dy, theta):
    return np.array([[ np.cos(theta), -np.sin(theta), dx],
                     [ np.sin(theta),  np.cos(theta), dy],
                     [             0,              0,  1]])


def cameraCoordinateComply(item2CameraPosition, cameraPoseFiltered):
    ''' AD-HOC COMPENSATION! DO NOT USE BLINDLY.
    Applies a set of modifications to each of the results so far so that
    the compound system coordinate frames are consistent.
    '''
    # Invert XZ directions so +X==Left & +Z==Front
    item2CameraPosition = [[bbox_class, -z, -x] for (bbox_class, z, x) in item2CameraPosition]

    # Redefine yaw as the horiz-plane clock-wise angle between Zcam & Zworld
    cameraPoseFiltered = [cameraPoseFiltered[0],
                          cameraPoseFiltered[1],
                          180 - cameraPoseFiltered[2]]

    return item2CameraPosition, cameraPoseFiltered


def relu(x):
    return np.max(x, 0)


def imagePatch(img, img_overlay, x, y):
    ''' Overlap a smaller image on top of a bigger one
    '''
    # compensate for patch radious so that x,y tell the center
    h, w = img_overlay.shape
    y -= h//2
    x -= w//2

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    # Add overlay within the determined ranges
    img[y1:y2, x1:x2] = img[y1:y2, x1:x2] + img_overlay[y1o:y2o, x1o:x2o]

    return np.clip(img, 0, 255)



def imagePatchTopLeftCorner(img, img_overlay, x, y):
    # DO NOT compensate for patch radious so that x,y tell the center
    #h, w, _ = img_overlay.shape
    #y -= h//2
    #x -= w//2

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    # Add overlay within the determined ranges
    # img[y1:y2, x1:x2] = img[y1:y2, x1:x2] + img_overlay[y1o:y2o, x1o:x2o]
    img[y1:y2, x1:x2] = img_overlay[y1o:y2o, x1o:x2o]

    return np.clip(img, 0, 255)


def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.
    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].

    This function was stolen from
    https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    return img




class worldMap(object):
    ''' Implements a world map class in which item detections are to be drawn
    '''
    def __init__(self, mapsize_XZ_cm=[320,320], items = []):
        # unpack map size (default is for bedroom)
        self.sizeZcm = mapsize_XZ_cm[0]
        self.sizeXcm = mapsize_XZ_cm[1]

        # infer how many items are we detecting and which are them
        if items != []:
            # count items
            self.numItems = len(items)

            # build a dictionary name-> index
            self.itemDict = {}
            for idx, name in enumerate(items):
                self.itemDict[name] = idx

            # build the opposite transformation as list index -> name
            self.itemNames = list(self.itemDict.keys())
        else:
            # Default configuration
            self.numItems = 1
            self.itemDict = {'': 0}
            self.itemNames = None


        # Map template
        self.map = np.zeros((self.sizeXcm, self.sizeZcm, self.numItems), np.float32)
        #
        # Coordinates of the World Frame with respect to the Map Frame
        self.world2map_XZ_cm = [110, 0]


        ### Init FoV triangle
        #
        # distance of view & angle aperture
        self.fov_dist = 300
        self.fov_ang  = np.deg2rad(25)
        #
        # define ficticious ABC with respect to the camera frame for FoV
        # These represent a triangular FoV in the horizontal plane
        self.a2camera = np.array([self.fov_dist,  self.fov_dist*np.tan(self.fov_ang), 1])
        self.b2camera = np.array([self.fov_dist, -self.fov_dist*np.tan(self.fov_ang), 1])
        self.c2camera = np.array([0,0, 1])
        #
        # Concatenate in matrix for quicker computations
        self.abc2camera = np.c_[self.a2camera, self.b2camera, self.c2camera]

        '''For debugging - remember that tripod yaw is not coaxial with camera Y
        self.a2camera = np.array([40,  0, 1])
        self.b2camera = np.array([ 30, 30, 1])
        self.c2camera = np.array([ 0,  0, 1])'''


        ### Init for detections
        #
        # GAUSSIAN KERNEL
        # Produce 2D gaussian kernel with top value close to white and shape (X,X,1)
        # Imshowing this kernel will not be representative, loot at detectionMask!
        ksize, sigma = [200, 20]
        kernel = cv2.getGaussianKernel(ksize=ksize,sigma=sigma)
        kernel = kernel @ kernel.T
        kernel *= 250/np.max(kernel)
        self.gaussianKernel2D = np.array([kernel], np.uint8).reshape(ksize,ksize,1)
        #
        # Speed at which items appear / disappear
        self.disappearanceRate = 0.003
        self.appearanceRate = 0.005


        ### Init fancy map_fov - load these variables from outside
        # image holding the map information
        self.map_background = []
        #
        # pixel coordinates of the origin of that map
        self.backgroundX = []
        self.backgroundY = []


    def getFoV_demo2world(self, cameraPose):
        # unpack coordinates for readability, mm-deg -> cm-rad
        x_cam   = 0.1 * cameraPose[0]
        z_cam   = 0.1 * cameraPose[1]
        yaw_cam = np.deg2rad(cameraPose[2])

        a2world = translationMatrix(z_cam, x_cam, yaw_cam) @ self.a2camera
        b2world = translationMatrix(z_cam, x_cam, yaw_cam) @ self.b2camera
        c2world = translationMatrix(z_cam, x_cam, yaw_cam) @ self.c2camera

        return a2world, b2world, c2world


    def getPoints_demo2world(self, item2CameraPositionCompliant, cameraPose):
        # Might be many points - parallelizable matrix implementation
        # unpack coordinates for readability, mm-deg -> cm-rad
        x_cam   = 0.1 * cameraPose[0]
        z_cam   = 0.1 * cameraPose[1]
        yaw_cam = np.deg2rad(cameraPose[2])

        # Remove class string and create a matrix in which each column is a position vector
        # we add a residual 1 to match matrix dimensions...
        # ...but we add as a 10 then divide the matrix by 10 to convert mm->cm
        p2camera = [i[1:] + [10] for i in item2CameraPositionCompliant]
        p2camera = np.array(p2camera).T/10

        # compute the translation matrix (common for all) and broadcast it along one dimension
        tm = translationMatrix(z_cam, x_cam, yaw_cam)
        # BROADCASTING NOT REQUIRED BUT LEFT COMMENTED FOR REFERENCE
        ##tm = np.repeat(tm[np.newaxis,:,:], len(item2CameraPositionCompliant), axis=0)

        # simultaneously transform the points in p2camera with the translation matrix
        # output vector contains a matrix in which each column is a point
        return tm @ p2camera



    def getFoV_2map(self, cameraPose):
        ''' Convert the coordinates of the ABC points that represent the FoV
        to map absolute coordinates.
        '''
        # unpack coordinates for readability, mm-deg -> cm-rad
        # Because WorldFrame and MapFrame are square to each other we add coords
        x_cam2map   = 0.1 * cameraPose[0]  + self.world2map_XZ_cm[0]
        z_cam2map   = 0.1 * cameraPose[1]  + self.world2map_XZ_cm[1]
        yaw_cam2map = np.deg2rad(cameraPose[2])

        # Apply homogeneous transformation matrix to the three points at once
        abc2map = translationMatrix(z_cam2map, x_cam2map, yaw_cam2map) @ self.abc2camera

        # Remove the tailing [1]s and swap coordinate place so that...
        # ...cv2 reads 3 different 2D points the way we want to. (ad-hoc)
        # output has the shape [[x1,z1], [x2,z2], [x3,z3]]
        abc2map = abc2map[[1,0],:].T

        return abc2map.astype(int)


    def getPoints_2map(self, item2CameraPosition, cameraPose):
        ''' Convert the coordinates of arbitrary points from camera
        to map absolute coordinates.
        '''
        # Might be many points - parallelizable matrix implementation
        # unpack coordinates for readability, mm-deg -> cm-rad
        x_cam2world  = 0.1 * cameraPose[0]
        z_cam2world   = 0.1 * cameraPose[1]
        yaw_cam2world = np.deg2rad(cameraPose[2])

        # Because WorldFrame and MapFrame are square to each other
        x_cam2map = x_cam2world + self.world2map_XZ_cm[0]
        z_cam2map = z_cam2world + self.world2map_XZ_cm[1]
        yaw_cam2map = yaw_cam2world

        # Remove class string and create matrix in which each column is a position vector
        # we append a residual [1] to match matrix dimensions later...
        # ...but we add as a [10] then divide the matrix by 10 to convert mm->cm
        p2camera = [i[1:] + [10] for i in item2CameraPosition]
        p2camera = np.array(p2camera).T / 10

        # simultaneously transform the points in p2camera with the translation matrix
        # output vector contains a matrix in which each column is a point
        p2map = translationMatrix(z_cam2map, x_cam2map, yaw_cam2map) @ p2camera

        # Remove the tailing [1]s and swap coordinate place so that...
        # ...cv2 reads 3 different 2D points the way we want to. (ad-hoc)
        p2map = p2map[[1,0],:].T

        # encode the category of each detection
        detectedCategories = [self.itemDict[i[0]] for i in item2CameraPosition]

        return p2map.astype(int), detectedCategories



    def update(self, item2CameraPosition, cameraPose):
        ''' Update existing likelihood map with any detection / missdetection
        '''

        # Get FoV mask (region of visible points)
        self.fovMask = np.zeros((self.sizeXcm, self.sizeZcm, self.numItems), np.float32)
        fovTriangle = self.getFoV_2map(cameraPose)
        cv2.drawContours(self.fovMask, [fovTriangle], 0, [255]*self.numItems, -1)

        # export fov triangle to self (poor planning ahead)
        self.fovTriangle = fovTriangle

        # Get discovery mask (regions where items are thought to be NOW)
        self.discoveryMask = np.zeros((self.sizeXcm, self.sizeZcm, self.numItems), np.uint16)
        if item2CameraPosition != []:
            item2MapPosition, categories = self.getPoints_2map(item2CameraPosition, cameraPose)
            for (x, y), category in zip(item2MapPosition, categories):
                layer = category
                self.discoveryMask[:,:,category] = \
                    imagePatch(self.discoveryMask[:,:,category], self.gaussianKernel2D[:,:,0], x, y)

        # Update likelihood map. Uniform decrease + gaussian increase. Clip for grayscale
        self.map = np.clip(
            self.map - self.fovMask * self.disappearanceRate
                     + self.discoveryMask * self.appearanceRate,
            0, 255)

        # this just to show fov over detections
        self.map_fov = self.map.copy()
        cv2.drawContours(self.map_fov, [fovTriangle], 0, [255]*self.numItems, 2)
        cv2.circle(self.map_fov, tuple(fovTriangle[2]), 20, [255]*self.numItems, 9)

        """
        likelihood maps are stored on the channels of self.map_fov an self.maps!!!
        """


    def getFancyMap(self):
        '''
        Use this to place the funky looking black and white likelihood map onto
        a background image to make it look more stylish.

        This is not efficient for many items but it works :)

        NOTE: This uses a transparent image! the transparent region is where the
        likelihood map will be placed.
        '''
        self.fancyDict = {}

        mapChannels = cv2.split(self.map)

        for i in range(self.numItems):
            # Linear invert the likelihood map to display black over white
            temp_map_fov = 255 - np.clip(mapChannels[i] * 255, 0,255)
            temp_map_fov = cv2.merge((temp_map_fov,temp_map_fov,temp_map_fov))

            # Volatile copy of background just for display
            background_temp = self.map_background[:,:,:3].copy()

            # Overlay map onto background
            self.fancyMap = imagePatchTopLeftCorner(background_temp, temp_map_fov, self.backgroundX, self.backgroundY)

            # Add Fov and stuff in colors
            coordChange = np.array([self.backgroundX, self.backgroundY])
            cv2.drawContours(self.fancyMap, [self.fovTriangle] + coordChange, 0, [150,50,0], 2)
            cv2.circle(self.fancyMap, tuple(self.fovTriangle[2] + coordChange), 20, [150,50,0], 9)

            # Smoothen likelihood map so it does not look so pixelated
            self.fancyMap = cv2.filter2D(self.fancyMap,-1,np.ones((3,3),np.float32)/9)

            # reoverlay background in the opposite direction to apply transparency mask
            alpha_mask  = self.map_background[:, :, 3] / 255.0
            img_result  = self.fancyMap[:, :, :3].copy()
            img_overlay = self.map_background[:, :, :3]
            outputFancyMap = overlay_image_alpha(img_result, img_overlay, 0, 0, alpha_mask)

            self.fancyDict[self.itemNames[i]] = outputFancyMap.copy()


        return self.fancyDict


    def packImages(self, additionalStuff, out_height=[]):
        ''' Build a single image with all we have so far. Provide maps as tuple'''

        # This catches trying to plot the maps when the likelihood maps are still
        # unitialized because the camera has not located its own position
        try:
            self.getFancyMap()
        except:
            return additionalStuff

        canvas = self.fancyDict[self.itemNames[0]]
        for idx, i in enumerate(self.itemNames):
            if idx == 0:
                continue
            else:
                canvas = np.hstack((canvas, self.fancyDict[i]))

        additionalStuff = cv2.resize(additionalStuff, (canvas.shape[0], canvas.shape[0]))
        canvas = np.hstack((canvas, additionalStuff))

        return cv2.pyrDown(canvas)




        """
        # make map a multilayer array with the number of different items to detect
        # map = np.zeros(x,z, numitems)

        # have a dict like this where the number is the layer in the map array:
        categories_dict = {'bottle': 0,
                           'can':    1,}

        # modify getPoints_2map with additional argument
        # so that if we detect bottle, can, can, bottle ->  detectedCategories = [0,1,1,0]

        def getPoints_2map(*args, *kwargs):
            ...
            detectedCategories = [categories_dict(i[0]) for i in item2CameraPosition]
            ...
            return p2map.astype(int), detectedCategories

        # then, in update, do:
        self.discoveryMask = np.zeros((self.sizeXcm, self.sizeZcm, numitems), np.uint16)
        if item2CameraPosition != []:
            item2MapPosition, detectedCategories = self.getPoints_2map(item2CameraPosition, cameraPose)
            for (x, y), cat in zip(item2MapPosition, detectedCategories):
                self.discoveryMask = imagePatch(self.discoveryMask[:,:, cat], self.gaussianKernel2D, x, y)

        # debug and profit
        """
