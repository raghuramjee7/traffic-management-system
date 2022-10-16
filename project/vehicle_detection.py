import cv2
import numpy as np
from time import sleep
from file_upload import *

class VehicleDetector():

    def __init__(self):
        self.car_count = 0
        return

    def detect(self, link):

        # ----- Predefined Variables ------
        width_min = 80
        height_min = 80 

        offset = 6 

        pos_line = 550

        delay = 60

        detect = []

        # ----- End of predefined variables ------

        	
        def center(x, y, w, h):
            x1 = int(w / 2)
            y1 = int(h / 2)
            cx = x + x1
            cy = y + y1
            return cx,cy

        cap = cv2.VideoCapture(link)
        subtract = cv2.createBackgroundSubtractorMOG2()

        while True:

            # ret if frame is avaiable, frame fetches the frame
            ret , frame1 = cap.read()
            temp = float(1/delay)
            sleep(temp)

            # convert the image to greyscale
            grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            #cv2.imshow("Video grey" , grey)

            # blur the image to remove extra greyscale
            blur = cv2.GaussianBlur(grey,(3,3),5)
            #cv2.imshow("Video after blur" , blur)

            # now lets use the blur image on backgroundsubtractor
            img_sub = subtract.apply(blur)
            #cv2.imshow("Video after mog2" , img_sub)

            
            # we dialate the image (similar to convolution) to find more features
            dilated = cv2.dilate(img_sub,np.ones((5,5)))

            # now we use erosion and dilution at the same time using morphologyex
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            diluted = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
            diluted = cv2.morphologyEx (diluted, cv2. MORPH_CLOSE , kernel)

            # contours are the curves joining all the points
            # each vehicle will have its own contour
            # we will pass the diluted image to the findContours function
            # parameters:
            #   1. image
            #   2. retrives all the possible contours from the image,
            #   3. no redundant points are stored in the np array
            contours,h = cv2.findContours(diluted,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            cv2.line(frame1, (25, pos_line), (1200, pos_line), (255,127,0), 3)
            
            for(i,c) in enumerate(contours):
                # get the rectangle bounding the vehicle
                # x coordinate, y coodinate, width, height
                (x,y,w,h) = cv2.boundingRect(c)

                # we set the minimum threshold, if a countour is less than that, we dont consider it
                valid_contours = (w >= width_min) and (h >= height_min)
                if not valid_contours:
                    continue

                # we outline the contour and the center
                cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
                v_center = center(x, y, w, h)
                detect.append(v_center)
                cv2.circle(frame1, v_center, 4, (0, 0,255), -1)

                # detecting vehicle once after it passes the position line
                for (x,y) in detect:
                    if y<(pos_line+offset) and y>(pos_line-offset):
                        self.car_count+=1
                        cv2.line(frame1, (25, pos_line), (1200, pos_line), (0,127,255), 3)  
                        detect.remove((x,y))
                        #print("Vehicles detected : "+str(self.car_count))  

            # show the car count in frame
            cv2.putText(frame1, "VEHICLE COUNT : "+str(self.car_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
            cv2.imshow("Video Original" , frame1)
            #cv2.imshow("Detector",diluted)

            # to exit if we press escape key
            if cv2.waitKey(1) == 27:
                break

        # once after we end, it closes all windows and video files
        cv2.destroyAllWindows()
        cap.release()
