import cv2
import numpy as np
import time

import pyautogui as gui
import pyautogui.tweens
gui.FAILSAFE = False

from collections import deque
from scipy import stats
# Function to find angle between two vectors

screenX, screenY = pyautogui.size()
frameX, frameY = 1000, 600

def angle(v1,v2):
    dot = np.dot(v1,v2)
    x_modulus = np.sqrt((v1*v1).sum())
    y_modulus = np.sqrt((v2*v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# Function to find distance between two points in a list of lists
def find_distance(A,B): 
    return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2))

def detect_face(image):
    faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces if len(faces) else None

def intersect(x_1, y_1, width_1, height_1, x_2, y_2, width_2, height_2):
    return not (x_1 > x_2+width_2 or x_1+width_1 < x_2 or y_1 > y_2+height_2 or y_1+height_1 < y_2)


def get_contour(frame, lower_bound, upper_bound):
    blur = cv2.blur(frame,(3,3))
    #Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    #Create a binary image with where white will be skin colors and rest is black
    #mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
    mask2 = cv2.inRange(hsv,np.array(lower_bound),np.array(upper_bound))
    #Kernel matrices for morphological transformation
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,127,255,0)

    #Find contours of the filtered frame
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    #cv2.imshow('Gesture',median)
    #k = cv2.waitKey(5) & 0xFF
    #if k == 27:
        #return None
    #return None
    #Draw Contours
    #Find Max contour area (Assume that hand is in the frame)
    if not len(contours):
        return None
    max_area=100
    ci=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            ci = i
        #Largest area contour
    cnts = contours[ci]
    return cnts

def augment_graph(frame, contour):
    if contour is None:
        return None, None
    moments = cv2.moments(contour)
    #Central mass of first order moments
    #if moments['m00']!=0:
    #    cx = int(moments['m10']/moments['m00']) # cx = M10/M00
    #    cy = int(moments['m01']/moments['m00']) # cy = M01/M00
    #centerMass = (cx,cy)
    #Draw center mass
    #circle
    (x,y),radius = cv2.minEnclosingCircle(contour)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(frame,center,7,[100,0,255],2)    
    cv2.circle(frame,center,radius,(92, 66, 244),5)
    return center, radius

def start_detect_hand(gesture_call_back=None):
    #Open Camera object
    cap = cv2.VideoCapture(0)
    #fire_img = cv2.imread('./fire.png')
    #Decrease frame size
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, frameX)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, frameY)    
    while(True):
    
        ret, frame = cap.read()
        #print frame[frame.shape[0]/2][frame.shape[1]/2]
        # skin color [2,50,50], [15,255,255]
        # pink color [160,50,160], [180,255,255]
        # green color [50, 100, 100], [70, 255, 255]
        pink_lower_bound = [160,50,160]
        pink_upper_bound = [180,255,255]
        yellow_lower_bound = [25, 115, 120]
        yellow_upper_bound = [90, 235, 255]
        
        pink_cnts = get_contour(frame=frame, lower_bound=pink_lower_bound, upper_bound=pink_upper_bound)
        yellow_cnts = get_contour(frame=frame, lower_bound=yellow_lower_bound, upper_bound=yellow_upper_bound)        
        
        
        pink_center_mass, pink_radius = augment_graph(frame=frame, contour=pink_cnts)
        yellow_center_mass, yellow_radius = augment_graph(frame=frame, contour=yellow_cnts)
        cv2.imshow('Gesture',frame)
        if gesture_call_back:
            gesture_call_back(pink_center_mass, pink_radius, yellow_center_mass, yellow_radius)
        #close the output video by pressing 'ESC'
        k = cv2.waitKey(2) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def move(previous_position, a, b):
    gui.moveTo((screenX - 1) - ((screenX - 1) * a / (frameX - 1)), (screenY - 1) * b / (frameY - 1), 0.06, pyautogui.easeInOutQuad)
    #pass

def is_decreased_sequence(sequence):
    for i in xrange(1, len(sequence)):
        if sequence[i] < sequence[i-1]:
            return False
    return True

def click():
    gui.click()
    #print 'click'

def double_click():
    pass

def collision_detect(x1,y1,r1,x2,y2,r2):
    return (x2-x1)**2 + (y1-y2)**2 <= (r1+r2)**2

previous_center = None
radius_queue = deque()
slope_queue = deque()

def gesture_call_back(pink_center, pink_radius, yellow_center, yellow_radius):
    global previous_center, radius_queue, slope_queue
    if previous_center is None:
        previous_center = pink_center
    elif pink_center is not None:
        old_x, old_y = previous_center
        x, y = pink_center
        distance = np.sqrt(np.power(x-old_x,2)+np.power(y-old_y,2))

        #if distance >= 200:
            #previous_center = center
            #return
        move(previous_center, x, y)
        previous_center = pink_center
        
    if pink_center and pink_radius and yellow_center and yellow_radius:
        if collision_detect(pink_center[0], pink_center[1], pink_radius, yellow_center[0], yellow_center[1], yellow_radius):
            click()

def main():
    start_detect_hand(gesture_call_back=gesture_call_back)
    
if __name__ == '__main__':
    main()