from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import numpy as np
import cv2

path_neural_net = 'data/neural_net/gestures_net.pkl'


# This method will read image from disk and convert to a 2D matrix
# of 0 and 1 and then flat the matrix to get a single array.
def read_and_transform_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    transformed_img = np.multiply(img, 1.0 / 255).flatten()
    return transformed_img


def load_images():
    x = []
    y = []
    print("Start loading images...")
    for i in range(1, 11):
        transformed_img = read_and_transform_image('data/train/1/1-' + str(i) + '.png')
        x.append(transformed_img)
        y.append(1)

    for i in range(1, 11):
        transformed_img = read_and_transform_image('data/train/2/2-' + str(i) + '.png')
        x.append(transformed_img)
        y.append(2)
    data = {"x": x, "y": y}
    print("Loading images finished")
    return data


def persist_neural_net(net, path):
    joblib.dump(net, path)


def load_neural_net(path):
    return joblib.load(path)


def train():
    data_set = load_images()
    x = data_set["x"]
    y = data_set["y"]
    print("Start training neural network...")
    # Classification http://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification
    # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    clf = MLPClassifier(verbose=True, solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 100), random_state=1)
    clf.fit(x, y)
    print("Finished training. Persisting trained neural net")
    persist_neural_net(clf, path_neural_net)
    print("Persisting done")


def init():
    global clf
    print("Loading persisted neural net")
    clf = load_neural_net(path_neural_net)
    print("Loading done")


def classify(image):
    # test_image1 = read_and_transform_image('data/train/1/1-2.png')
    # test_image2 = read_and_transform_image('data/train/2/2-3.png')
    # prediction = clf.predict([test_image1])
    flat_image = image.flatten()
    prediction_prob = clf.predict_proba([flat_image])
    print("prediction_prob: ", prediction_prob)
    max_index = np.argmax(prediction_prob)
    max_class = clf.classes_[max_index]
    max_prob = prediction_prob[0][max_index]
    return {
        "class": max_class,
        "prob": max_prob
    }
