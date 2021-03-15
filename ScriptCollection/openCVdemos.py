import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

image = cv2.imread('cocacola1.jpg')

'''primeiro tutorial ''' # INSTALLING AND FIRST PLAYING OPENCV

def demo1(image = image):       # open, saving, showing
    """Abrir, gardar, cerrar imagen"""

    cv2.imshow('hello_world', image)
    print(image.shape)
    # cv2.imwrite('filename', image) # para gardar a imaxe
    cv2.waitKey()
    cv2.destroyAllWindows()

def demo2(image = image):       # img to black&white
    """Pasa a imagen a blanco e negro"""

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('hello_world', gray_image)
    print(gray_image.shape)
    cv2.waitKey()

    B,G,R = image[0,0]  # devolve o color do primeiro pixel
    print("Primeiro pixel B G R:", B,G,R)
    # print(image.shape)  # dimensions of image
    # print(gray_image.shape)
    print("Primeiro pixel BW:", gray_image[0,0])     # color do primeiro pixel

def demo3(image = image):       # img to HSV
    """Convirte a imagen a HSV (hue saturation value)"""

    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    cv2.imshow('HSV image',hsv_image)
    cv2.imshow('Hue channel',hsv_image[:,:,0])
    cv2.imshow('saturation channel',hsv_image[:,:,1])
    cv2.imshow('value channel',hsv_image[:,:,2])
    cv2.waitKey()
    cv2.destroyAllWindows()

def demo4(image = image):       # split RGB components
    ''' Segmenta en RGB; O orden dos colores non é exactamten o que se
        espera, non lle dei moita importancia inda

        Despois de segmentar plotea en cada color, en escala de grises,
        Canto mais valor dese color teña o rgb mais blanco, claro'''

    B,G,R = cv2.split(image)
    cv2.imshow("Red",R)
    cv2.imshow("Green",G)
    cv2.imshow("Blue",B)
    merged=cv2.merge([B,G,R])
    cv2.imshow("merged",merged)

    merged=cv2.merge([B+100,G,R]) #amplifying the blue color
    cv2.imshow("merged with blue amplify",merged)

    print(B.shape)
    print(R.shape)
    print(G.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo5(image = image):       # split rgb and print individual components
    ''' Segmento en colores e pidolle que me devolva as fotos en cadanseu color pai'''

    B,G,R=cv2.split(image)
    zeros=np.zeros(image.shape[:2],dtype="uint8")
    cv2.imshow("RED",cv2.merge([zeros,zeros,R]))
    cv2.imshow("Green",cv2.merge([zeros,G,zeros]))
    cv2.imshow("Blue",cv2.merge([B,zeros,zeros]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    cv2.imshow("hsbcapada",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo6(image = image):       # plot histograms for RGB pixels of each colour
    '''Pinta histogramas cos colores: cantos pixeles de cada color hai con cada valor'''
    histogram = cv2.calcHist([image], [0], None, [256], [0,256])
    plt.hist(image.ravel(), 256, [0,256]) #o ravel parece substituir a linea de arriba
    plt.show()

    color = ('b', 'g', 'r')
    for i, col in enumerate(color):         ### MAZO INTERESANTE METODO ENUMERATE()
        histogram2 = cv2.calcHist([image], [i], None, [256], [0,256])
        plt.plot(histogram2, color = col)
        plt.xlim([0,256])
        plt.show()

    '''
    cv2.calcHist([images], channels, mask, histsize, ranges):

    > images:   Source image, must be given between []
    > channels: Also inside []; index channel for which histogram is calculated:
            == [0] for grayscale
            == [1] for blue
            == [2] for green
            == [3] for red
    > mask:     Mask image. To get hist for full image input None
                If a particular region is to be histogrammed this is done here
    > histsize: BIN count, for full scale we pass [256]
    > range:    normally [0,256]
        '''

def demo7():                    # drawing various shapes
    ''' Examples of free drawing blank image'''

    '''Create blanks'''
    image = np.zeros((512,512,3), np.uint8) # creates a numpy array loaded with zeros (black image)
    # cv2.imshow("black rectangle(color)", image)

    image_bw = np.zeros((512,512), np.uint8) # no RGB just 1 channel
    # cv2.imshow("black rectangle(BW)", image_bw)

    '''Create lines'''
    # cv2.line(host_image, starting coordinates, ending coordinates, color, thickness)
    cv2.line(image, (0,0), (511,511), (255,127,0), 5)
    # cv2.imshow("blue line", image)

    '''Create a rectangle'''
    # cv2.rectangle(host_image, starting coordinates, ending coordinates, color, thickness)
    cv2.rectangle(image, (30,50), (100,150), (255,127,0), 5)
    # cv2.imshow("rectangle", image)

    '''Create a circle'''
    cv2.circle(image, (100,100), (50), (255,127,0), -1)
    # cv2.imshow("circle", image)

    '''Create polyline'''
    pts = np.array( [[10,50], [400,60], [30,89], [90,68]], np.int32 ) # points for the polyline
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image, [pts], True, (0,255,255), 3)
    # cv2.imshow("polygon", image)

    '''Text'''
    # cv2.putText(image,'text to display',bootom left starting point, font,font size, color, thickness)
    cv2.putText(image, "hello world", (75,290), cv2.FONT_HERSHEY_COMPLEX, 2, (100,170,0), 3)

    cv2.imshow("hello world", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


''' Segundo tutorial empeza desde aqui ''' # IMAGE MANIPULATIONS IN OPENCV PT1
'''
    https://circuitdigest.com/tutorial/image-manipulation-in-python-opencv-part1
    Affine and Non-Affine Image transformations:
    > Affine:
            Scaling, rotation, translation
            Parallel lines remain parallel
    > Non-affine:   (or projective/homography)
            Not affine

command->cv2.warpAffine()

'''

def demo8(image = image):       # translation
    ''' Image translation '''

    height, width = image.shape[:2]     # store image dimensions
    print(image.shape[:2])

    '''
    requires Translation matrix to define movement direction
        T = [1  0   Tx; 0   1   Ty]
    '''

    quater_height, quater_width  =  height/4, width/4   # will move by one quarter
    T = np.float32([[1, 0, quater_width], [0, 1, quater_height]]) #translation matrix

    img_translation = cv2.warpAffine(image, T, (width, height))
    print(T)

    cv2.imshow("original", image)
    cv2.waitKey(0)

    cv2.imshow("Translation", img_translation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo9(image = image):       # rotation
    ''' Image rotation

    requires Rotation matrix to define movement direction
        M = [cosG  -sinG; sinG   -sinG] / G == rotation angle counterclockwise+

    In-built userfriendly method to rotate&scale:
        cv2.getRotationMatrix2D(rotation_center_x, rotation_center_y,
            angle of  rotation, scale)

    Transposing the image would also work for 90º lefty turn
    '''

    height, width = image.shape[:2]  # will rotate about center, divide these /2
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)

    rotate_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    cv2.imshow('original image', image)
    cv2.waitKey(0)
    cv2.imshow('rotated image', rotate_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo10(image = image):      # scaling resizing interpolation
    '''Scaling, resizing, interpolation

    > Interpolation:  generating new pixels within a discrete
                    set of known data points
    > Some methods:
        cv2.INTER_AREA      - good for shrinking / downsampling
        cv2.INTER_NEAREST   - fastest
        cv2.LINEAR          - good for zooming / upsampling
        cv2.CUBIC           - better
        cv2.INTER_LANCZOS4  - best

        LINEAR INTERPOLATION IS SET BY DEFAULT '''


    # cv2.resize(image,dsize(output image size), x_scale, y_scale, interpolation)

    image_scaled = cv2.resize(image, None, fx = 0.75, fy = 0.75)
    cv2.imshow("scaled image", image_scaled)
    cv2.waitKey(0)

    img_double = cv2.resize(image_scaled, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    cv2.imshow("x2 zoom", img_double)
    cv2.waitKey(0)

    # exact dimensions rescaling
    image_resize=cv2.resize(image,(200,300),interpolation=cv2.INTER_AREA)
    cv2.imshow('scaling_exact',image_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo11(image = image):      # pyramid
    '''Image pyramids: direct scaling in halves/doubles'''
    smaller = cv2.pyrDown(image)
    larger = cv2.pyrUp(smaller)

    cv2.imshow('original',image)
    cv2.waitKey(0)
    cv2.imshow('smaller',smaller)
    cv2.waitKey(0)
    cv2.imshow('larger',larger)
    cv2.waitKey(0)

    # increase quality of larger using cubic interpolation
    img_double = cv2.resize(smaller, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    cv2.imshow("interpolated", img_double)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo12(image = image):      # cropping
    '''Cropping: not cv2 method, just matrix cropping '''
    # cropped = image[start_row : end_row, start_col : end_col]

    height, width=image.shape[:2]
    start_row, start_col = int(height*.25),int(width*.25)
    end_row, end_col = int(height*.75),int(width*.75)

    cropped = image[start_row:end_row, start_col:end_col]
    cv2.imshow("original image",image)
    cv2.waitKey(0)
    cv2.imshow("cropped image", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo13(image = image):      # brightening/darkening
    ''' Brightening and darkening with arithmetic operations'''

    # gives a matrix with same dimension as image with all values = 100
    M = np.ones(image.shape, dtype="uint8") * 100

    # adding this matrix to the image powers up all RGB components by 100
    added = cv2.add(image, M)
    cv2.imshow("Added",added)
    cv2.waitKey(0)

    # removing this matrix to the image powers down all RGB components by 100
    subtracted = cv2.subtract(image, M)
    cv2.imshow("subtracted",subtracted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


''' Terceiro tutorial empeza desde aqui ''' # IMAGE MANIPULATIONS IN OPENCV PT2
''' https://circuitdigest.com/tutorial/image-manipulation-in-python-opencv-part2'''

def demo14():                               # bitwise operations
    '''Bitwise operations and masking'''
    # create square and ellipse
    square = np.zeros((300,300), np.uint8) # only 2D because its BW
    cv2.rectangle(square, (50,50), (250,250), 255, -1)
    cv2.imshow("square", square)


    ellipse=np.zeros((300,300),np.uint8)
    cv2.ellipse(ellipse,(150,150),(150,150),30,0,180,255,-1)
    cv2.imshow("ellipse",ellipse)


    # we can now apply logic to these shapes, which are binary data
    # same as this way, i can kill parts of real images with bitwise
    BitwiseAND = cv2.bitwise_and(square, ellipse)
    BitwiseOR = cv2.bitwise_or(square,ellipse)
    BitwiseXOR = cv2.bitwise_xor(square,ellipse)
    BitwiseNOT_elp = cv2.bitwise_not(ellipse)

    cv2.imshow("BitwiseAND", BitwiseAND)
    cv2.imshow("BitwiseOR", BitwiseOR)
    cv2.imshow("BitwiseXOR", BitwiseXOR)
    cv2.imshow("BitwiseNOT_elp", BitwiseNOT_elp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo15(image = cv2.pyrDown(image)):     # Convolution and blurring
    ''' Convolution and blurring'''
    ''' When convoluting, we define a kernel matrix which represents
        the size of the pixel-box where operations happen

        Image convoluting is an element wise multiplication of two
        matrices followed by a sum

        Blurring: averaging kernel pixels
        '''

    cv2.imshow("original, imported pyrdown", image)

    # create kernel as matrix which sums 1 (hence the /9), else we modify brigthness
    kernel_3x3 = np.ones((3, 3), np.float32) /9
    kernel_7x7 = np.ones((7, 7), np.float32) /49

    blurred3 = cv2.filter2D(image, -1, kernel_3x3)
    blurred7 = cv2.filter2D(image, -1, kernel_7x7)

    cv2.imshow("3x3 blurred", blurred3)
    cv2.imshow("7x7 blurred", blurred7)

    cv2.waitKey()
    cv2.destroyAllWindows()

    ''' other blurring methods:

    > cv2.blur: convolve the image with normalized box filter, this takes the
              place under the box and replaces the central element. Box size
              odd and positive
    > cv2.GaussianBlur
    > cv2.medianBlur: replaces central element with median of kernel
    > cv2.bilateralFilter: ueful for noise removal but keeping edges up   '''

    cv2.imshow("Averaging with cv2.blur",   cv2.blur(image,(3, 3)))
    cv2.imshow("Gaussian blurring",         cv2.GaussianBlur(image, (7,7),0))
    cv2.imshow("Median Blur",               cv2.medianBlur(image, 5))
    cv2.imshow("Bilateral blurring",        cv2.bilateralFilter(image, 9, 75, 75))

    cv2.waitKey()
    cv2.destroyAllWindows()

    ''' Denoising non local means denoising

    cv2.fastNlMeansDenoising() – for single gray scale image
    cv2.fastNlMeansDenoisingColored() – Single color image
    cv2.fastNlmeansDenoisingMulti() – for image sequence grayscale
    cv2.fastNlmeansDenoisingcoloredMulti() – for image sequence colored
    cv2.imshow("original, pyrdown", image)  '''

    #parameter after None is the filter strength 'h'(5-10 is a good range)
    # next is h for color components, set as same value as h again
    dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
    cv2.imshow('Fast means denois',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo16(image = cv2.pyrDown(image)):     # Sharpening
    ''' Sharpening '''
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    cv2.imshow("sharpened",     cv2.filter2D(image, -1, kernel))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo17(image = cv2.imread("image_gradiente.JPG")):          # Thresholding
    ''' Thresholding: converting an image into binary form 1/0
                      NEEDS BW-LIKE IMAGE INPUT (monochannel)
    Types...
        cv2.THRESH_BINARY       – set threshold, above goes white below goes black
        cv2.THRESH_BINARY_INV   – set threshold, above goes black below goes white
        cv2.THRESH_TRUNC        - set threshold, above goes thresh-value below goes black
        cv2.THRESH_TOZERO       - set threshold, below goes to zero rest unchanged
        cv2.THRESH_TOZERO_INV   - set threshold, above goes to zero rest unchanged
    '''
    # cv2.threshold(image, threshold value, Max value, threshold type)
    cv2.imshow('original',image) # gradient image to apply threshold to
    cv2.waitKey(0)

    # '_' para ignorar valores ao desempaquetar
    # value below 127 go to 0 (black), and above 127 goes to 255(white)
    _,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('1 threshold',thresh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #value below 127 goes to 255 and values above 127 goes to 0(reverse of above) (THRESH_BINARY_INV)
    _,thresh2=cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('2 threshold',thresh2)
    cv2.waitKey(0)

    #value above 127 are truncated (held) at 127, the 255 argument is unused.
    _,thresh3=cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
    cv2.imshow('3 thresh trunc', thresh3)
    cv2.waitKey(0)

    #values below 127 goes to 0, above 127 are unchanged
    _,thresh4=cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
    cv2.imshow('4 threshold', thresh4)
    cv2.waitKey(0)

    #Revesrse of above, below 127 is unchanged, above 127 goes to zero
    _,thresh5=cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)
    cv2.imshow('5 threshold', thresh5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo18(image = cv2.imread("image_dilationerosion.JPG")):    # Dilation and erosion
    ''' Dilation, Erosion, Opening/Closing

    Dilation: add pixels to boundaries of objects
    Erosion: removes pixels at object boundaries
    Opening: erosion then dilation                  - helpful for noise cleaning
    Closing: dilation then erosion

    Erosion and dilatin work in reverse for images with white background
    (By default black color is considered background)
    '''

    cv2.imshow('original', image)
    cv2.waitKey(0)

    kernel=np.ones((5,5),np.uint8)

    # Erosion, N iterations means repeating for N times (successively removes boundary layers)
    erosion = cv2.erode(image, kernel, iterations = 1)
    cv2.imshow('Erosion', erosion)
    cv2.waitKey(0)

    # Dilation
    dilation = cv2.dilate(image, kernel, iterations = 1)
    cv2.imshow('dilation', dilation)
    cv2.waitKey(0)

    # opening, good for removing the noise
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imshow('opening', opening)
    cv2.waitKey(0)

    # closing, Good for removing noise
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closing', closing)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def demo19(image = cv2.pyrDown(image)):     # edge detection
    '''Edge detection and image gradients'''
    '''
    Edge detection algorithms:
        > Sobel: emphasis on vertical/horizontal images
        > Laplacian: optimal due to low error rate, well defined edges and accurate detection
        > Canny Edge detection algorithm:
            - Applies Gaussian blur
            - Finds intensity gradient of image
            - Applies non-maximum suppression (remove pixels that are not edges)
            - Hysteresis applies threshold (if pixel within upper and lower threshold -> edge)
    '''


    height,width = image.shape[:2]

    # Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    cv2.imshow('original',image)
    cv2.waitKey(0)

    cv2.imshow('sobelx',sobel_x)
    cv2.imshow('sobely',sobel_y)
    cv2.waitKey(0)

    sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
    cv2.imshow('sobelOR',sobel_OR)
    cv2.waitKey(0)

    # Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    cv2.imshow('Laplacian', laplacian)
    cv2.waitKey(0)

    #canny edge detection algorithm uses gradient values as thresholds
    #in canny we need to provide two values: threshold1 and threshold2.
    #any gradient larger than threshold 2 is considered to be an edge.
    #any gradient larger than threshold 1 is considered not to be an edge.
    #values in between threshold 1 and threshold 2 are either as edge or non-edge
    #on how their intensities are connected, in this case any value below 60 are considered
    #non edges wheareas any value above 120 are considered as edges.

    canny = cv2.Canny(image, 60, 120)
    cv2.imshow('canny', canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo20():                               # perspective and affine transformation
    ''' Perspective & Affine transform '''

    ''' Perspective transform '''
    image = cv2.imread('image_perspective.JPG')
    cv2.imshow('original',image)
    cv2.waitKey(0)

    #coordinate of 4 corners of original image
    points_A=np.float32([[219,9],[494,160],[60,410],[380,523]])

    #coordinates of 4 corners of desired output
    #we use a ratio of an A4 paper 1:1.41
    points_B=np.float32([[0,0],[420,0],[0,592],[420,592]])

    M = cv2.getPerspectiveTransform(points_A, points_B) # gets transformation matrix from points
    warped = cv2.warpPerspective(image, M, (420,594))
    cv2.imshow('warpprespective',warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ''' Affine transform -> only need 3 points
        three points can represent: scale in X
                                    scale in Y
                                    rotation
    '''

    image=cv2.imread('image_affine.JPG')
    rows, cols = image.shape[:2]
    cv2.imshow('original', image)
    cv2.waitKey(0)

    #coordinate of 3 corners of original image
    points_A=np.float32([[320,15],[700,215],[85,610]])

    #coordinates of 3 corners of desired output
    #we use a ratio of an A4 paper 1:1.41
    points_B=np.float32([[0,0],[420,0],[0,592]])

    #use the two sets of two points to compute the Affine
    #transformation matrix,M
    M=cv2.getAffineTransform(points_A, points_B)
    warped=cv2.warpAffine(image, M, (cols,rows))
    cv2.imshow('warpaffine', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


''' Project at the end of the 3rd tutorial '''

def _sketch(image):              # default alternative to be called from while-loop
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # image to BW
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5),0) # Gaussian blurring
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)

    ret, mask = cv2.threshold(canny_edges,70,255,cv2.THRESH_BINARY_INV)
    return mask

def _identifyred(image):         # self-made toy for filtering red
    # B,G,R = cv2.split(image)
    # BW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # BitwiseAND = cv2.bitwise_and(square, ellipse)

    # image = cv2.GaussianBlur(image, (5,5),0) # Gaussian blurring

    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_frame = cv2.medianBlur(hsv_frame, 5)


    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(image, image, mask=red_mask)

    red = cv2.medianBlur(red, 5)


    return red

def live_feed():                 # live loop for webcam capturing and calling previous funcs
    cap=cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('livesketcher', _sketch(frame))
        if cv2.waitKey(20) == 13:    #13 is the enterkey
            break

    #release camera and close window, remember to release the webcam with the help of cap.release()
    cap.release()
    cv2.destroyAllWindows()

# live_feed() # function call




''' TUTORIAL 4: ''' # IMAGE SEGMENTATION USING OPENCV
'''https://circuitdigest.com/tutorial/image-segmentation-using-opencv'''

def demo21():       # segmentation & contours
    ''' Segmentation by contours technique '''
    image = cv2.imread('squares.jpg')
    cv2.imshow('input image', image)
    cv2.waitKey(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # to grayscale

    edged = cv2.Canny(gray, 30, 200)    # canny edges
    cv2.imshow('canny edges', edged)
    cv2.waitKey(0)

    ''' Approximation methods are used for not storing all contour points, but simpler data,
    such as start-end points. Two cases below. Also, we use copy bc contours alters orinal'''
    #contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # THESE MAY REQUIRE A FIRST _, ASSIGNMENT TARGET DEPENDING ON THE VERSION


    cv2.imshow('canny edges after contouring', edged)
    cv2.waitKey(0)

    print(hierarchy,'\n') # ?? small integer matrix
    print(contours)
    print('Numbers of contours found =', str(len(contours))) # returns 3, one closed contour per square


    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    cv2.imshow('contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo22():       # hierarchy & retrieval mode
    '''
    Retrieval mode defines the hierarchy in contours like sub contours or external contours
    Four retrievas modes:
        cv2.RETR_LIST       - retrieves all contours
        cv2.RETR_EXTERNAL   - retrieves external/outer contours
        cv2.RETR_CCOMP      - retrieves all in a 2-level hierarchy
        cv2.RETR_TREE       - retrieves all in full hierarchy
    '''
    # RETR_LIST && RETR_EXTERNAL

    image = cv2.imread('squares_donut.jpg')
    cv2.imshow('input image', image)
    cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray, 30, 200)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_l, hierarchy_l = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print('Numbers of contours found=', len(contours), '&', len(contours_l)) # 8???? nonsense?


    cv2.drawContours(image,contours,-1,(0,255,0),3) #use -1 as the 3rd parameter to draw all the contours
    cv2.imshow('EXTERNAL',image)

    cv2.drawContours(image,contours_l,-1,(0,255,0),3) #use -1 as the 3rd parameter to draw all the contours
    cv2.imshow('LIST',image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo23():       # approximating contours

    '''
    cv2.approxPolyDP(contour, approximation accuracy, closed)
        Contour                 -  individual contour we want to approximate
        Approximation accuracy  -  reccommended <5% contour perimeter
        Closed                  -  bool if contour needs is open/closed
    '''
    image = cv2.imread('drawhouse.jpg')
    orig_image = image.copy()
    cv2.imshow('original image', orig_image)
    cv2.waitKey(0)

    # Grayscale & binarize image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # iterate through each contour and compute its BOUNDING RECTANGLE
    for c in contours:
        x, y, w, h = cv2.boundingRect(c) # origin X & Y, width and height
        cv2.rectangle(orig_image, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.imshow('Bounding rect', orig_image)
    cv2.waitKey(0)

    # iterate through each contour and compute its APPROX CONTOUR
    for c in contours:
        accuracy = 0.01 * cv2.arcLength(c, True)        # 3% of each contour's perimeter
        approx = cv2.approxPolyDP(c, accuracy, True)
        cv2.drawContours(image, [approx], 0, (0,255,0), 2)
        cv2.imshow('Approx polyDP', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo24():       # convex hull: smallest polygon that can fit around an object
    image = cv2.imread('star.jpg')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 176, 255, 0)

    contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    #Sort contours by area and remove largest frame contour
    n = len(contours) - 1
    contours = sorted(contours, key = cv2.contourArea, reverse = False)[:n] # simply -1??

    image2 = image.copy()
    cv2.drawContours(image2, contours, -1, (0,255,0), 2)
    cv2.imshow('plain contours',image2)

    # iterate contours and draw convex hull
    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(image, [hull], 0, (0,255,0), 2)
    cv2.imshow('convex hull', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo25():       # matching contour by shapes (find elements with given template contour)
    '''
    cv2.matchShapes(contour template, contour, method, method parameter)
        contour template    -   reference contour we try to find
        contour             -   individual contour we are checking against
        method              -   type of contour matching
        method parameter    -   leave alone as 0.0 in python distributions
    '''

    template = cv2.imread('star_lonely.jpg')
    cv2.imshow('template', template)

    target = cv2.imread('shapestomatch.jpg')
    cv2.imshow('target', target)
    cv2.waitKey(0)

    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(template, 127, 255, 0)
    ret, thresh2 = cv2.threshold(gray, 127, 255, 0)

    contours, hierarhy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #we need to sort the contours by area so we can remove the largest contour which is
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #we extract the second largest contour which will be our template contour
    template_contour = contours[1]

    #extract the contours from the second target image
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


    for c in contours:
        #iterate through each contour in the target image and use cv2.matchShape to compare the contour shape
        match = cv2.matchShapes(template_contour, c, 1, 0.0) # similarity degree
        print("match")
        if match < 0.16:
            closest_contour = c

    cv2.drawContours(target, [closest_contour], -1, (0,255,0), 3)
    cv2.imshow('output',target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo26():       # identifying shapes
    ''' PINTASE A FIGURA RECHEA PORQUE ESTA ESPECIFICADO O PARAMETRO THICKNESS COMO -1, DESPOIS DE COLOR'''
    image = cv2.imread('shapes.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('identifying shapes', image)
    cv2.waitKey(0)

    ret, thresh = cv2.threshold(gray, 127, 255, 1)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


    for cnt in contours:

        # busca aproximar os contornos a poligonos simples
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)

        if len(approx) == 3:
            shape_name="Triangle"
            cv2.drawContours(image, [cnt], 0, (0,255,0), -1)

            # CALCULATE CENTROID OF THE BLOB (shape)
            M = cv2.moments(cnt)
            cx=int(M['m10'] /M['m00'])
            cy=int(M['m01'] /M['m00'])

            cv2.putText(image, shape_name, (cx-50,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)

        elif len(approx)==4:
            x,y,w,h=cv2.boundingRect(cnt)

            M=cv2.moments(cnt)
            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])

            #cv2.boundingRect returns the left width and height in pixels, starting...
            #       ...from the top left corner, for square it would be roughly same

            if abs(w-h) <= 5:           # if width ~= height ==> square
                shape_name = "square"
                cv2.drawContours(image, [cnt], 0, (0,125,255), -1)
                cv2.putText(image, shape_name, (cx-50,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)

            else:
                shape_name = "Reactangle"
                cv2.drawContours(image, [cnt], 0, (0,0,255), -1)
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(image,shape_name,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)

        elif len(approx)==10:
            shape_name='star'
            cv2.drawContours(image, [cnt], 0, (255,255,0), -1)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(image,shape_name,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)

        elif len(approx)>=15:
            shape_name = 'circle'
            cv2.drawContours(image, [cnt], 0, (0,255,255), -1)
            M = cv2.moments(cnt)
            cx=int(M['m10'] / M['m00'])
            cy=int(M['m01'] / M['m00'])
            cv2.putText(image,shape_name,(cx-50,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
        cv2.imshow('identifying shapes', image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo27():       # line detection
    ''' Rectas en open cv representadas en polar respecto a origen da  foto, por linea normal p, phi

    cv2.HoughLines(binarized image, ρ accuracy, Ө accuracy, threshold)
        threshold   -   minimum value for something to be considered a line
    '''
    # HOUGH

    image = cv2.imread('linedetect.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 170, apertureSize = 3)
    cv2.imshow('edges', edges)


    #theta accuracy of (np.pi / 180) which is 1 degree
    #line threshold is set to 240 (number of points on line)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 240)

    #we iterate through each line and convert into the format
    #required by cv2.lines(i.e. requiring end points)
    for i in range(0, len(lines)):
        for rho, theta in lines[i]:
            a=np.cos(theta)
            b=np.sin(theta)
            x0=a*rho
            y0=b*rho
            x1=int(x0+1000*(-b))
            y1=int(y0+1000*(a))
            x2=int(x0-1000*(-b))
            y2=int(y0-1000*(a))
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow('hough lines',image)
    cv2.waitKey(0)



    # PROBABILISTIC HOUGH
    image=cv2.imread('linedetect.jpg')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,50,150,apertureSize=3)
    #again we use the same rho and theta accuracies
    #however, we specify a minimum vote(pts along line) of 100
    #and min line length of 5 pixels and max gap between the lines of 10 pixels

    lines=cv2.HoughLinesP(edges,1,np.pi/180,100,100,10)
    for i in range(0,len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)

    cv2.imshow('probalistic hough lines',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def demo28():   # blob detection
    '''
    blob == collection of connected pixels which share a property
    cv2.drawKeypoints(input image, keypoints, blank_output_array, color, flags)

    where in the flags could be
        cv2.DRAW_MATCHES_FLAGS_DEFAULT
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
        cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

    blank == 1x1 matrix of 0s

    '''
    image = cv2.imread('sunflowers.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(image)

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensure the...
    #  ...size of circle corresponds to the size of blob
    blank = np.zeros((1,1))
    blobs = cv2.drawKeypoints(image,keypoints,blank,(0,255,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    cv2.imshow('blobs', blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo29():   # blob filtering
    '''
    cv2.SimpleBlobDetector_Params()

    We will see filtering the blobs by mainly these four parameters listed below:
    Area
        params.filterByArea=True/False
        params.minArea=pixels
        params.maxArea=pixels
    Circularity
        params.filterByCircularity=True/False
        params.minCircularity=  1 being perfect, 0 being opposite
    Convexity  - Area of blob/area of convex hull
        params.filterByConvexity= True/False
        params.minConvexity=Area
    Inertia
        params.filterByInertia=True/False
        params.minInertiaRatio=0.01
    '''
