import numpy as np
import cv2

print(cv2.__version__)


def flip_frame(frame_to_flip):
    return cv2.flip(frame_to_flip, 1)


# Giving file name will read from video file instead from camera
capture = cv2.VideoCapture(0)
# Removing background https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html
fgbg = cv2.createBackgroundSubtractorMOG2(history=50,
                                          detectShadows=True)  # detectShadows=True can be added as a parameter
initial = True
# Define the codec and create VideoWriter object to save video to file
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
c=0
while True:
    if capture.isOpened():
        # Capture frame by frame
        ret, frame = capture.read()
        if ret:
            width = round(capture.get(3))
            height = round(capture.get(4))
            # print('width'+str(width))
            # print('height'+str(height))
            flippedFrame = flip_frame(frame)

            frameWithoutBackground = fgbg.apply(flippedFrame)
            gray = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(cv2.medianBlur(frameWithoutBackground, 11), 0, 1,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            if initial:
                merged_frame = np.ones([height, width], np.uint8)
                initial = False
            print("-----"+str(c))

            print(merged_frame)

            if c > 0:
                merged_frame = np.bitwise_and(merged_frame, thresh1)
                print("++++-----")
                print(thresh1)
                print("after")
                print(merged_frame)
            c=c+1
            cv2.imshow('frame1', np.multiply(merged_frame, 255))
            cv2.imshow('frame2', np.multiply(thresh1, 255))

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
