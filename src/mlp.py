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


def classify(image):
    print("Loading persisted neural net")
    clf = load_neural_net(path_neural_net)
    print("Loading done")
    test_image1 = read_and_transform_image('data/train/1/1-2.png')
    # test_image2 = read_and_transform_image('data/train/2/2-3.png')
    # prediction = clf.predict([test_image1])
    prediction_prob = clf.predict_proba([test_image1])
    max_index = np.argmax(prediction_prob)
    max_class = clf.classes_[max_index]
    max_prob = prediction_prob[0][max_index]
    print(max_class)
    print(max_prob)
    return {
        "class": max_class,
        "prob": max_prob
    }



# x = [[0., 0.,
#       1., 0.,
#       1., 1.],
#      [1., 0.,
#       1., 0.,
#       1., 0.],
#      [0., 0.,
#       0., 1.,
#       0., 1.],
#      [0., 1.,
#       0., 1.,
#       0., 1.]]
# y = [0, 0, 1, 1]
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(x, y)
# MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#               beta_1=0.9, beta_2=0.999, early_stopping=False,
#               epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
#               learning_rate_init=0.001, max_iter=200, momentum=0.9,
#               nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#               solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#               warm_start=False)
# prediction = clf.predict([[0., 0.,
#                            1., 0.,
#                            1., 1.],
#                           [1., 0.,
#                            1., 0.,
#                            1., 0.],
#                           [1., 1.,
#                            0., 1.,
#                            0., 1.]])
# print(prediction)

# train()
classify("")