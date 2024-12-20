import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn import neighbors
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression

def load_data(dataset_name):
    if "sphere" in dataset_name:
        print("loading sphere dataset")
        mat = loadmat("data/sphere.mat")
        X = mat["X_hds"]
        Y = mat["t"]
        return X, Y
    elif "mnist" in dataset_name:
        print("loading MNIST dataset")
        mat = loadmat("data/MNIST.mat")
        X = mat["X_hds"]
        Y = mat["t"]
        return X, Y
    else:
        print("loading COIL20 dataset")
        mat = loadmat("data/COIL_20.mat")
        X = mat["X_hds"]
        Y = mat["t"]
        return X, Y

# NlogN computational complexity: can be a moderately slow
def tSNE_visualisation(X, Y, perplexity = 30): # neighbour embedding: similarity preservation
    Xld = TSNE(2, init='pca', learning_rate='auto', perplexity = perplexity).fit_transform(X)
    plt.scatter(Xld[:, 0], Xld[:, 1], c = Y)
    plt.show()

# NxN computation complexity: can be very slow
def MDS_visualisation(X, Y): # metric multidimensional scaling: preservation of the distances
    Xld = MDS(n_components=2).fit_transform(X)
    plt.scatter(Xld[:, 0], Xld[:, 1], c = Y)
    plt.show()

def PCA_visualisation(X, Y):
    Xld = PCA(n_components=2).fit_transform(X)
    plt.scatter(Xld[:, 0], Xld[:, 1], c = Y)
    plt.show()

def run_ex1(X, Y):
    N, M = X.shape

    print("ex 1, dataset shape:", N, M)

def run():
    # X, Y = load_data("sphere")
    X, Y = load_data("mnist")
    # X, Y = load_data("sphere")

    # tSNE_visualisation(X, Y, perplexity = 20)
    PCA_visualisation(X, Y)
    # MDS_visualisation(X, Y)

    run_ex1(X, Y)

if __name__ == "__main__":
    run()
