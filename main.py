import cv2
import numpy as np

cap=cv2.VideoCapture('vid.mp4')

bg=cv2.imread("bg1.jpg")

while True:
    rt, frame= cap.read()
    frame=cv2.pyrDown(frame)
    
   
    Hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bg=cv2.resize(bg, (frame.shape[1],frame.shape[0]), interpolation=cv2.INTER_AREA)

    
    mask0=cv2.inRange(Hsv, np.array([95, 150, 20]), np.array([120, 255, 255]))

    mask0=cv2.dilate(mask0, np.ones((5,5)), iterations=4)
    mask0=cv2.erode(mask0, np.ones((5,5)), iterations=4)

    mask1= cv2.bitwise_not(mask0)

    masked0=cv2.bitwise_and(bg, bg, mask=mask0)
    masked1=cv2.bitwise_and(frame, frame, mask=mask1)

    final=cv2.addWeighted(masked0, 1, masked1, 1, 0)

    cv2.imshow("output", final)
    cv2.waitKey(10)


cap.release()
cv2.destroyAllWindows()