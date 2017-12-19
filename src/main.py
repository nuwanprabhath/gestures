import numpy as np
import cv2
import mlp
from skimage.measure import compare_ssim as ssim

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
c = 0


# Structural Similarity https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def structural_similarity(image_1, image_2):
    return ssim(image_1, image_2)


def get_gesture_from_category(cat):
    if cat == 1:
        return "Left swipe"
    elif cat == 2:
        return "Right swipe"
    elif cat == 3:
        return "Swipe Down"
    elif cat == 4:
        return "Shrink"


progress = 1
previous_deviation = 0
# fileName = 11
while True:
    if capture.isOpened():
        # Capture frame by frame
        ret, frame = capture.read()
        if ret:
            width = round(capture.get(3))
            height = round(capture.get(4))
            # print('width'+str(width)) #width 1280
            # print('height'+str(height)) #height 720
            # print(capture.get(cv2.CAP_PROP_FPS))
            flippedFrame = flip_frame(frame)

            frameWithoutBackground = fgbg.apply(flippedFrame)
            gray = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(cv2.medianBlur(frameWithoutBackground, 11), 0, 255,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            if initial:
                merged_frame = np.ones([height, width], np.uint8)
                merged_frame = np.multiply(merged_frame, 255)
                initial_frame = merged_frame
                initial = False
            text = ""
            if c > 10:
                # merged_frame = np.bitwise_and(merged_frame, np.multiply(thresh1, 0.25))
                # merged_frame = np.multiply(thresh1, progress)
                thresh1[thresh1 == 0] = progress
                mod = np.mod(thresh1, 255)
                add = np.add(merged_frame, mod)
                merged_frame = add
                print('-------')
                # cv2.imshow('Capture', add)
                deviation = structural_similarity(initial_frame, merged_frame)
                print("deviation")
                print(deviation)
                text = ""
                # print("previous_deviation")
                # print(previous_deviation)
                # print("substact")
                # print(abs(deviation - previous_deviation))
                # print("prog")
                # print(progress)
                if progress > 255:
                    progress = 1

                if abs(deviation - previous_deviation) > 0.001:
                    progress = progress + 10
                previous_deviation = deviation
                if deviation < 0.8:
                    # cv2.imwrite('5-'+str(fileName)+'.png', add)
                    # fileName = fileName+1
                    print("---------")
                    classification = mlp.classify(merged_frame)
                    category = classification["class"]
                    prob = classification["prob"]
                    print("Prob: " + str(prob))
                    print("Category: " + str(category))
                    if prob > 0.8:
                        text = get_gesture_from_category(category)
                        print("Found gesture: " + str(category))
                    else:
                        print("Not found")
                    print("================== Resetting image ==================")
                    merged_frame = np.ones([height, width], np.uint8)
                    merged_frame = np.multiply(merged_frame, 255)
                    previous_deviation = 0
                    progress = 1

            c = c + 1
            # blackAndWhiteImage = np.multiply(merged_frame, 255)
            # print(blackAndWhiteImage)
            cv2.imshow('Capture', cv2.putText(merged_frame, text, (540, 50),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                # cv2.imwrite('1-4.png', add)
                break
        else:
            break
    else:
        capture.open()

# Release the capture
capture.release()
# out.release()
cv2.destroyAllWindows()
