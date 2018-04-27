import numpy as np
import cv2

def show_image():
    intubot='/home/pi/intubot/interface/intubot.JPG'
    img = cv2.imread(intubot) # (intubot,0) 0 return a grayscale image
    #img=cv2.resize(img,(500,500))
    cv2.imshow('intubot',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def play_videos():
    video='/home/pi/intubot/intubation.mp4'
    cap = cv2.VideoCapture(video)

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',gray) # play the gray color video.
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

play_videos()
#show_image()
