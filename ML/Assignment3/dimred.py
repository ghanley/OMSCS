from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn import random_projection
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from sklearn.datasets import load_digits
from scipy import linalg
import itertools
from sklearn import mixture
import matplotlib as mpl
from yellowbrick.cluster import KElbowVisualizer
from numpy import linalg as LA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as ld


# Breast Cancer
data = pd.read_csv('breast.csv')
b_labels = data['Class']

data = data.values
n_samples, breast_n_features = data.shape
breast_labels = 2
breast_data = scale(data)


#Digits
data, d_labels = load_digits(return_X_y=True)
model = KMeans()
digits_data = scale(data)
n_samples, digits_n_features = data.shape
digits_labels = len(np.unique(d_labels))


# pca = PCA(n_components=20)
# pca2_results = pca.fit_transform(data)
# print(pca.explained_variance_ratio_.sum())
# ica = FastICA(n_components=20)
# ica2_results = ica.fit_transform(data)
# print(ica.explained_variance_ratio_.sum())

#Start PCA
digits_components = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,digits_n_features]
breast_components = [2,3,4,5,6,7,8,9,10,15,20,25,30, breast_n_features]
d_variance_record = []
b_variance_record = []

for i in digits_components:
    d_pca = PCA(n_components=i)
    d_pca2_results = d_pca.fit_transform(digits_data)
    d_variance_record.append(d_pca.explained_variance_ratio_.sum())


for i in breast_components:
    b_pca = PCA(n_components=i)
    b_pca2_results = b_pca.fit_transform(breast_data)
    b_variance_record.append(b_pca.explained_variance_ratio_.sum())

plt.clf()
plt.figure()
plt.title("PCA Explained Variance by Component #")
plt.plot(d_variance_record,'r-', alpha=0.7, label="Digits")
plt.plot(b_variance_record,'b-', alpha=0.7, label="Breast")
ticks = range(len(digits_components))

plt.xticks(ticks, digits_components)
# plt.xlim([-1, len(d_variance_record)])
plt.legend(loc="best")
plt.savefig('PCA.png')

# ica = FastICA(n_components=20)
# ica2_results = ica.fit_transform(data)
# print(ica.explained_variance_ratio_.sum())

#Start ICA
digits_components = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,digits_n_features]
breast_components = [2,3,4,5,6,7,8,9,10,15,20,25,30, breast_n_features]
d_variance_record = []
b_variance_record = []

for i in digits_components:
    d_pca = FastICA(n_components=i)
    temp = d_pca.fit_transform(digits_data)
    temp = pd.DataFrame(temp)
    temp = temp.kurt(axis=0)
    d_variance_record.append(temp.abs().mean())

for i in breast_components:
    b_pca = FastICA(n_components=i)
    temp = b_pca.fit_transform(breast_data)
    temp = pd.DataFrame(temp)
    temp = temp.kurt(axis=0)
    b_variance_record.append(temp.abs().mean())

plt.clf()
plt.figure()
plt.title("ICA Kurtosis by Component #")
plt.plot(d_variance_record,'r-', alpha=0.7, label="Digits")
plt.plot(b_variance_record,'b-', alpha=0.7, label="Breast")
ticks = range(len(digits_components))

plt.xticks(ticks, digits_components)
# plt.xlim([-1, len(d_variance_record)])
plt.legend(loc="best")
plt.savefig('ICA.png')


#Randomized Projections
digits_components = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,digits_n_features]
breast_components = [2,3,4,5,6,7,8,9,10,15,20,25,30, breast_n_features]
d_rp_error = []
b_rp_error = []

for i in digits_components:
    RP = random_projection.GaussianRandomProjection(n_components=i)
    rp2_results = RP.fit(digits_data)
    W = rp2_results.components_
    p = LA.pinv(W)
    reconstructed = ((p@W)@(digits_data.T)).T
    errors = np.square(digits_data - reconstructed)
    d_rp_error.append(np.nanmean(errors))


for i in breast_components:
    RP = random_projection.GaussianRandomProjection(n_components=i)
    rp2_results = RP.fit(breast_data)
    W = rp2_results.components_
    p = LA.pinv(W)
    reconstructed = ((p@W)@(breast_data.T)).T
    errors = np.square(breast_data - reconstructed)
    b_rp_error.append(np.nanmean(errors))

plt.clf()
plt.figure()
plt.title("Randomized Projection Reconstruction Error by Component")
plt.plot(d_rp_error,'r-', alpha=0.7, label="Digits")
plt.plot(b_rp_error,'b-', alpha=0.7, label="Breast")
ticks = range(len(digits_components))

plt.xticks(ticks, digits_components)
# plt.xlim([-1, len(d_variance_record)])
plt.legend(loc="best")
plt.savefig('rp.png')


#start LinearDiscriminantAnalysis
digits_components = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,digits_n_features]
breast_components = [2,3,4,5,6,7,8,9,10,15,20,25,30, breast_n_features]
d_variance_record = []
b_variance_record = []

for i in digits_components:
    d_lda = ld(n_components=i)
    d_lda2_results = d_lda.fit_transform(digits_data, d_labels)
    d_variance_record.append(d_lda.explained_variance_ratio_.sum())


for i in breast_components:
    b_lda = ld(n_components=i)
    b_lda2_results = b_lda.fit_transform(breast_data, b_labels)
    b_variance_record.append(b_lda.explained_variance_ratio_.sum())

plt.clf()
plt.figure()
plt.title("Linear Discriminate Analysis Explained Variance by Component #")
plt.plot(d_variance_record,'r-', alpha=0.7, label="Digits")
plt.plot(b_variance_record,'b-', alpha=0.7, label="Breast")
ticks = range(len(digits_components))

plt.xticks(ticks, digits_components)
# plt.xlim([-1, len(d_variance_record)])
plt.legend(loc="best")
plt.savefig('lda.png')
