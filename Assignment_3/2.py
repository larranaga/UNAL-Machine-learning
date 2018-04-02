from sklearn.datasets import fetch_mldata
from sklearn import preprocessing, svm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def print_number(arr):
    pixels = arr.reshape((28, 28))
    plt.imshow(pixels, cmap="gray")
    plt.show()


mnist = fetch_mldata('MNIST original')
data = np.array(mnist.data)
labels = np.array(mnist.target)

aux_number = np.array(data[0:1, 0:784])

digits = np.zeros(10, dtype=int)
for lavel in mnist.target.astype(int):
    digits[lavel] += 1

#print(digits)

tuples = list(zip(labels.astype(int), data))

d = {}
for key, val in tuples:
    d.setdefault(key, []).append(val)

#chosse digits 1 and 7
# trainig data set 80 % of samples
# test data set 20 % of samples
#print(len(d[1]))
#print(len(d[7]))
p_train = 0.8
p_test = 1 - p_train

ones_train_samples = np.ceil(len(d[1]) * p_train).astype(int)
sevens_train_samples = np.ceil(len(d[7]) * p_train).astype(int)

ones_test_samples = len(d[1]) - ones_train_samples
sevens_test_samples = len(d[7]) - sevens_train_samples

ones_train = d[1][:ones_train_samples]
sevens_train = d[7][:sevens_train_samples]

ones_test = d[1][-ones_test_samples:]
sevens_test = d[7][-sevens_test_samples:]
#print(np.asarray(ones_train).shape)
#print(np.asarray(sevens_train).shape)

training_data = preprocessing.scale(ones_train + sevens_train)
test_data = preprocessing.scale(ones_test + sevens_test)

training_target = [1] * ones_train_samples + [7] * sevens_train_samples
test_target = [1] * ones_test_samples + [7] * sevens_test_samples

histogram = {}
#space = [2**-15, 2**-10, 2**-5, 2**0, 2**5, 2**10, 2**15]

space = np.linspace(0.0000000000001, 1, 100)
for c in space:
    clf = svm.LinearSVC(C=c)
    clf.fit(training_data, training_target)
    histogram[c] = clf.score(test_data, test_target)
print(histogram)
""" plt.bar(range(len(histogram)), list(histogram.values()), align='center')
plt.xticks(range(len(histogram)), list(histogram.keys()))
plt.show() """