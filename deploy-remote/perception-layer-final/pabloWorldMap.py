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
    # compensate for patch radious so that x,y tell the center
    h, w, _ = img_overlay.shape
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



class worldMap(object):
    def __init__(self, mapsize_XZ_cm=[320,320], numItems=1):
        # unpack map size (default is for bedroom)
        self.sizeZcm = mapsize_XZ_cm[0]
        self.sizeXcm = mapsize_XZ_cm[1]

        # Map template
        self.map = np.zeros((self.sizeXcm, self.sizeZcm, numItems), np.float32)

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

        # GAUSSIAN KERNEL
        # Produce 2D gaussian kernel with top value close to white and shape (X,X,1)
        # Imshowing this kernel will not be representative, loot at detectionMask!
        ksize, sigma = [200, 20]
        kernel = cv2.getGaussianKernel(ksize=ksize,sigma=sigma)
        kernel = kernel @ kernel.T
        kernel *= 250/np.max(kernel)
        self.gaussianKernel2D = np.array([kernel], np.uint8).reshape(ksize,ksize,1)

        # Speed at which items appear / disappear
        self.disappearanceRate = 0.003
        self.appearanceRate = 0.005



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

        return p2map.astype(int)


    def update(self, item2CameraPosition, cameraPose):

        # Get FoV mask (region of visible points)
        self.fovMask = np.zeros((self.sizeXcm, self.sizeZcm, 1), np.float32)
        fovTriangle = self.getFoV_2map(cameraPose)
        cv2.drawContours(self.fovMask, [fovTriangle], 0, 255, -1)


        # Get discovery mask (regions where items are thought to be NOW)
        self.discoveryMask = np.zeros((self.sizeXcm, self.sizeZcm, 1), np.uint16)
        if item2CameraPosition != []:
            item2MapPosition = self.getPoints_2map(item2CameraPosition, cameraPose)
            for (x, y) in item2MapPosition:
                self.discoveryMask = imagePatch(self.discoveryMask, self.gaussianKernel2D, x, y)


        # Update likelihood map. Uniform decrease + gaussian increase. Clip for grayscale
        self.map = np.clip(
            self.map - self.fovMask * self.disappearanceRate
                     + self.discoveryMask * self.appearanceRate,
            0, 255)


        '''cv2.imshow('Field of view', self.fovMask)
        cv2.imshow('Instantaneous discovery', np.array(self.discoveryMask, np.uint8))
        cv2.imshow('Likelihood map', self.map)
        cv2.waitKey(1)
        '''



        #print(xz_item2world[0])
        #self.map = cv2.circle(self.map, (int(i) for i in xz_item2world[0]), 20, (255), -1)

        # convert continuous to discrete coordinates
        # x_i = np.digitize(x, self.x_bins)
        # z_i = np.digitize(z, self.z_bins)

        # self.map[z_i][x_i] += 0.5
        # decrease probability for false negatives that are recognized now
        #self.map = self.map - self.mask_removal
        # bound values
        #self.map = np.clip(self.map, 0, 2)
