import numpy as np
import cv2
import mlp
print(cv2.__version__)


def flip_frame(frame_to_flip):
    return cv2.flip(frame_to_flip, 1)


# Initialize neural net
mlp.init()
# Giving file name will read from video file instead from camera
capture = cv2.VideoCapture(0)
# Removing background https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html
fgbg = cv2.createBackgroundSubtractorMOG2(history=50,
                                          detectShadows=True)  # detectShadows=True can be added as a parameter
initial = True
# Define the codec and create VideoWriter object to save video to file
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
c = 0

# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
#
# # Change thresholds
# params.minThreshold = 10;
# params.maxThreshold = 200;
#
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 200
# #
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
#
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87
# #
# # # Filter by Inertia
# # params.filterByInertia = True
# # params.minInertiaRatio = 0.01
# detector = cv2.SimpleBlobDetector_create(params)


# Mean square error https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def mse(image_1, image_2):
    err = np.sum((image_1.astype("float") - image_2.astype("float")) ** 2)
    err /= float(image_1.shape[0] * image_1.shape[1])
    return err


while True:
    if capture.isOpened():
        # Capture frame by frame
        ret, frame = capture.read()
        if ret:
            width = round(capture.get(3))
            height = round(capture.get(4))
            # print('width'+str(width)) #width 1280
            # print('height'+str(height)) #height 720
            flippedFrame = flip_frame(frame)

            frameWithoutBackground = fgbg.apply(flippedFrame)
            gray = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(cv2.medianBlur(frameWithoutBackground, 11), 0, 1,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            if initial:
                merged_frame = np.ones([height, width], np.uint8)
                initial_frame = merged_frame
                initial = False

            if c > 10:
                merged_frame = np.bitwise_and(merged_frame, thresh1)
                deviation = mse(initial_frame, merged_frame)
                print("---------")
                print(deviation)
                if deviation > 0.3:
                    classification = mlp.classify(merged_frame)
                    category = classification["class"]
                    prob = classification["prob"]
                    print("Prob: " + str(prob))
                    print("Category: " + str(category))
                    if prob > 0.8:
                        print("Found gesture: " + str(category))
                    else:
                        print("Not found")
                    print("Resetting image")
                    merged_frame = np.ones([height, width], np.uint8)

            c = c + 1
            blackAndWhiteImage = np.multiply(merged_frame, 255)
            cv2.imshow('frame1', blackAndWhiteImage)
            # cv2.imshow('frame2', np.multiply(thresh1, 255))
            # overlay = thresh1.copy()
            # keypoints = detector.detect(thresh1)
            # print(keypoints)
            # for k in keypoints:
            #     print(k)
            #     cv2.circle(overlay, (int(k.pt[0]), int(k.pt[1])), int(k.size/2), (0, 0, 255), -1)
            #     cv2.line(overlay, (int(k.pt[0])-20, int(k.pt[1])), (int(k.pt[0])+20, int(k.pt[1])), (0,0,0), 3)
            #     cv2.line(overlay, (int(k.pt[0]), int(k.pt[1])-20), (int(k.pt[0]), int(k.pt[1])+20), (0,0,0), 3)
            #
            #     opacity = 0.5
            #     cv2.addWeighted(overlay, opacity, thresh1, 1 - opacity, 0, thresh1)
            # im_with_keypoints = cv2.drawKeypoints(flippedFrame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # print(keypoints)
            # cv2.imshow('frame2', im_with_keypoints)
            # cv2.imshow('frame3', thresh1)q

            if cv2.waitKey(25) & 0xFF == ord('q'):
                # cv2.imwrite('3-1.png', blackAndWhiteImage)
                break
        else:
            break
    else:
        capture.open()

# Release the capture
capture.release()
# out.release()
cv2.destroyAllWindows()
