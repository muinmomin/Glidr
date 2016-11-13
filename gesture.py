import cv2
import numpy as np
import time

#Open Camera object
cap = cv2.VideoCapture(0)

#Decrease frame size
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)

# Function to find angle between two vectors
def Angle(v1,v2):
    dot = np.dot(v1,v2)
    x_modulus = np.sqrt((v1*v1).sum())
    y_modulus = np.sqrt((v2*v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# Function to find distance between two points in a list of lists
def FindDistance(A,B): 
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


while(True):

    #Capture frames from the camera
    ret, frame = cap.read()
    faces = detect_face(image=frame)

    blur = cv2.blur(frame,(3,3))

    #Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))

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

    #Draw Contours
    #cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
    #cv2.imshow('temp',median)

    #Find Max contour area (Assume that hand is in the frame)
    if not len(contours):
        continue
    max_area=5000
    ci=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if faces is not None and len(faces):
                for (x,y,w,h) in faces:
                    if x < cX < x + w and y < cY < y + h:
                        continue
                    else:
                        max_area = area
                        ci = i
                        break
            else:
                max_area = area
                ci = i
        #Largest area contour
    cnts = contours[ci]
    
    #Find convex hull
    hull = cv2.convexHull(cnts)

    #Find convex defects
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)

    #Get defect points and draw them in the original image
    if defects is None:
        continue
    
    FarDefect = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        FarDefect.append(far)
        cv2.line(frame,start,end,[0,255,0],1)
        cv2.circle(frame,far,10,[100,255,255],3)

        #Find moments of the largest contour
    moments = cv2.moments(cnts)

    #Central mass of first order moments
    if moments['m00']!=0:
        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
        cy = int(moments['m01']/moments['m00']) # cy = M01/M00
    centerMass = (cx,cy)    

    #Draw center mass
    cv2.circle(frame,centerMass,7,[100,0,255],2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)     

    #Distance from each finger defect(finger webbing) to the center mass
    distanceBetweenDefectsToCenter = []
    
    for i in range(0,len(FarDefect)):
        x =  np.array(FarDefect[i])
        centerMass = np.array(centerMass)
        distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
        distanceBetweenDefectsToCenter.append(distance)

    #Get an average of three shortest distances from finger webbing to center mass
    sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
    AverageDefectDistance = np.mean(sortedDefectsDistances[:2])

    #Get fingertip points from contour hull
    #If points are in proximity of 80 pixels, consider as a single point in the group
    finger = []
    for i in range(0,len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
            if hull[i][0][1] <500 :
                finger.append(hull[i][0])

    #The fingertip points are 5 hull points with largest y coordinates  
    finger =  sorted(finger,key=lambda x: x[1])   
    fingers = finger[0:5]

    #Calculate distance of each finger tip to the center mass
    fingerDistance = []
    for i in range(0,len(fingers)):
        distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
        fingerDistance.append(distance)

    #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
    #than the distance of average finger webbing to center mass by 130 pixels
    result = 0
    for i in range(0,len(fingers)):
        if fingerDistance[i] > AverageDefectDistance+130:
            result = result +1

    #Print number of pointed fingers
    cv2.putText(frame,str(result),(100,100),font,2,(255,255,255),2)

    #show height raised fingers
    #cv2.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)

    #Print bounding rectangle
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.drawContours(frame,[hull],-1,(255,255,255),2)

    cv2.imshow('Dilation',frame)


    #close the output video by pressing 'ESC'
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()