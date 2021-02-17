from pynput.mouse import Button, Controller
from pynput import keyboard
import pyscreenshot as ImageGrab
import numpy as np
import cv2
import time

''' Pablo Santana - 12/02/2021

This script will record a section of your screen (that you have to select) and
will try to filter out a .png file with one of the items of the darpa dataset.

Meant to be used from this website:
https://app.ignitionrobotics.org/OpenRobotics/fuel/models/Vent

> Interrupt the program with ^C at any time
> Use a pixel with background color as upper-left corner of the screenshot region
> Create an empty folder named ./data in the directory you're working to hold pics

Please be aware that some keypresses may hold on to the buffer and show up in
the terminal in between stages. Try not writing any bash command while the
machine is asking you to press buttons and do not rush when running the program.
'''

## THIS SECTION TO SELECT PART OF THE SCREEN ----------------------------------

print('''Select boundary box coordinates for screen capture!
Hover your mouse over the desired place & press:
'1': Upper left corner
'2': Lower right corner
's': To save changes
'^C': To quit''')
x1, y1 = 0,0
x2, y2 = 0,0

def on_press(key):
    ''' This func is slave to the listener class. The actions to perform on
    keypresses are defined here. '''
    global x1, y1, x2, y2
    try:
        if key.char == '1':
            x1, y1 = x, y
        if key.char == '2':
            x2, y2 = x, y
        if key.char == 's':
            return False
    except:
        pass

# Init threads for mouse & keyboard tracking
mouse = Controller()
listener = keyboard.Listener(
    on_press=on_press)
listener.start()

# Keep printing current & goal coordinates
while listener.running:
    x, y = mouse.position
    print(f'>> Current {x}, {y}; UL {x1}, {y1}; LR {x2}, {y2}\r', end='\r')



## THIS SECTION TO TAKE A PIC & PROCESS THE IMAGE -----------------------------

# Name the class & init the counter
item = input('>> What item are we about to record?')
i = 0

# Will keep takin screenshots until ^C
while 1:
    i += 1

    # Capture a region of the screen based on these coordinates
    pic = np.array(ImageGrab.grab(bbox=[x1, y1, x2, y2]))

    def crop(pic, binary):
        '''This function uses a binary map input to compute its edges and crop a
        picture pic to those edges coordinates in a square shape.'''
        contours,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        return pic[y:y+h,x:x+w]

    # get a mask with the background area
    background_val = pic[0][0]
    lower = np.array([background_val], dtype="uint8")
    upper = np.array([background_val], dtype="uint8")
    mask = cv2.bitwise_not(cv2.inRange(pic, lower, upper))

    # apply mask to obtain black background
    pic = cv2.bitwise_and(pic,pic,mask = mask)

    # Detect edges in the mask and use them to crop the original picture
    pic = crop(pic, mask)

    # Detect edges in the mask and use them to crop itself
    mask = crop(mask, mask)

    # Add the mask as an alpha channel
    pic = np.dstack([pic, mask])

    # Export trimmed picture
    cv2.imwrite(f'./data/{item}_{i}.png', pic)

    # Msg & Awaiter loop so that pictures are taken only if the item has moved
    print(f'Move your mouse to keep taking pics! ({i} taken)')
    x, y = mouse.position
    while abs(x - mouse.position[0]) < 5:
        pass
