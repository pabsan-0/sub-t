import cv2

# Opens the Video file
cap = cv2.VideoCapture('./Subt_2.mp4')

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if i%(round(25*0.3)) == 0:
        print(i)
        if ret == False:
            break
        cv2.imwrite('./output/cave2-'+str(i)+'.jpg',frame)
    i+=1


cap.release()
cv2.destroyAllWindows()
