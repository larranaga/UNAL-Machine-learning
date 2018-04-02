from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
import numpy as np
import csv
mnist = fetch_mldata('MNIST original')
data = np.array(mnist.data).astype(float)
with open("scaledData.csv", "w") as f:
    writer = csv.writer(f)
    for row in data:
        rs = row.reshape(1, 784)
        ps = preprocessing.scale(rs[0])
        writer.writerow(ps)
""" for row in data:
    rs = row.reshape(1, 784)
    ps = preprocessing.scale(rs[0])
    print(ps) """
