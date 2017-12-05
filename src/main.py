import numpy as np
import cv2;

print(cv2.__version__);


def flip_frame(frame_to_flip):
    return cv2.flip(frame_to_flip, 1)


# Giving file name will read from video file instead from camera
capture = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object to save video to file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
while True:
    if capture.isOpened():
        # Capture frame by frame
        ret, frame = capture.read()
        if ret:
            # print(capture.get(3)) Width
            # print(capture.get(4)) Height
            flippedFrame = flip_frame(frame)
            # Operations on the frame come here
            gray = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2GRAY)
            # thresholdImage = cv2.adaptiveThreshold(cv2.medianBlur(gray, 11), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # Less noise
            thresholdImage1 = cv2.adaptiveThreshold(cv2.medianBlur(gray, 11), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            # thresholdImage = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (11, 11), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            # thresholdImage = cv2.adaptiveThreshold(cv2.bilateralFilter(gray,5,75,75), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            # Write to file
            # out.write(flippedFrame)
            # Display resulting frame
            # cv2.imshow('frame', thresholdImage)
            cv2.imshow('frame1', thresholdImage1)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    else:
        capture.open()

# Release the capture
capture.release()
# out.release()
cv2.destroyAllWindows()
