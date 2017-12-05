import cv2;

print(cv2.__version__);
capture = cv2.VideoCapture(0)

while True:
    if capture.isOpened():
        # Capture frame by frame
        ret, frame = capture.read()

        # Operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.open()

# Release the capture
capture.release()
cv2.destroyAllWindows()
