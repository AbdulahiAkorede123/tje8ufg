import cv2
body_classifier=cv2.CascadeClassifier("haarcascade_fullbody.xml")


# Create our body classifier


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies=body_classifier.detectMultiScale(grey,1.2,3)
    # Pass frame to our body classifier
    
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("output",frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
