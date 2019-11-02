#code adapated from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

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
from sklearn.neural_network import MLPClassifier
import util
from collections import defaultdict
from time import clock


np.random.seed(42)

# # Breast Cancer
# data = pd.read_csv('breast.csv')
# labels = data['Class']
# data = data.values
# data = scale(data)
# n_samples, n_features = data.shape
# n_digits = 2

# #Digits
# data, labels = load_digits(return_X_y=True)
# data = scale(data)
# n_samples, n_features = data.shape
# n_digits = len(np.unique(labels))



iterations = [2,3,4, 5,6,7,8,9, 10, 15, 20,25,30, 100]
sample_size = 300

def run_cluster(data, labels, n_samples, n_features, n_digits, title, run_extra=False):

  train_score = defaultdict(list)
  train_score['k-means'] = []
  train_score['gmm'] = []

  for i in iterations:
    n_digits = i

    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=1)
    kmeans.fit_transform(data)
    cluster_feature = kmeans.labels_
    train_score['k-means'].append(metrics.v_measure_score(labels, kmeans.labels_))

    gaus = GaussianMixture(n_components=n_digits)
    gaus.fit(data)
    cluster_feature2 = gaus.predict(data)
    train_score['gmm'].append(gaus.bic(data))
    # print('pre', data.shape)
    # print('pre1', cluster_feature.shape)
    if run_extra:
      clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
      timings[title]['default'] = 0
      start = clock()
      util.plot_learning_curve(clf, title+" Basic ANN LC", data, labels, cv=5, n_jobs=-1)
      timings[title]['default'] += clock() - start


    data1 = np.column_stack((data, cluster_feature))
    # print('post1', data.shape)
    # print('EM', cluster_feature2.shape)
    data2 = np.column_stack((data, cluster_feature2))

    if run_extra:
      clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
      timings[title]['KmeansXtra'] = 0
      start = clock()
      util.plot_learning_curve(clf, title+" KM Cluster Feature ANN LC", data1, labels, cv=5, n_jobs=-1)
      timings[title]['KmeansXtra'] += clock() - start

      clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
      timings[title]['EMxtra'] = 0
      start = clock()
      util.plot_learning_curve(clf, title+" EM Cluster Feature ANN LC", data2, labels, cv=5, n_jobs=-1)
      timings[title]['EMxtra'] += clock() - start



  plt.figure()
  ticks = range(len(iterations))

  plt.plot(ticks, train_score['k-means'], 'o-', label="Train Score", color="g" )
  plt.xticks(ticks, iterations)
  plt.title(title+"K Means V-Measure Score")
  plt.xlabel("N Clusters")
  plt.ylabel("V Measure")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.savefig(title+'KmeansVmeasure.png')

  plt.figure()
  ticks = range(len(iterations))
  plt.plot(ticks, train_score['gmm'], 'o-', label="Train Score", color="g" )
  plt.xticks(ticks, iterations)
  plt.title(title+"EM BIC Score")
  plt.xlabel("N Clusters")
  plt.ylabel("BIC")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.savefig(title + 'EMbic.png')


  # lowest_bic = np.infty
  # bic = []
  # n_components_range = range(1, 7)
  # X = data
  # cv_types = ['spherical', 'tied', 'diag', 'full']
  # for cv_type in cv_types:
  #     for n_components in n_components_range:
  #         # Fit a Gaussian mixture with EM
  #         gmm = mixture.GaussianMixture(n_components=n_components,
  #                                       covariance_type=cv_type)
  #         gmm.fit(X)
  #         bic.append(gmm.bic(X))
  #         if bic[-1] < lowest_bic:
  #             lowest_bic = bic[-1]
  #             best_gmm = gmm

  # bic = np.array(bic)
  # color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
  #                               'darkorange'])
  # clf = best_gmm
  # bars = []

  # # Plot the BIC scores
  # plt.figure(figsize=(8, 6))
  # spl = plt.subplot(2, 1, 1)
  # for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
  #     xpos = np.array(n_components_range) + .2 * (i - 2)
  #     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
  #                                   (i + 1) * len(n_components_range)],
  #                         width=.2, color=color))
  # plt.xticks(n_components_range)
  # plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
  # plt.title('BIC score per model')
  # xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
  #     .2 * np.floor(bic.argmin() / len(n_components_range))
  # plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
  # spl.set_xlabel('Number of components')
  # spl.legend([b[0] for b in bars], cv_types)

  # # Plot the winner
  # splot = plt.subplot(2, 1, 2)
  # Y_ = clf.predict(X)
  # for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
  #                                           color_iter)):
  #     v, w = linalg.eigh(cov)
  #     if not np.any(Y_ == i):
  #         continue
  #     plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

  #     # Plot an ellipse to show the Gaussian component
  #     angle = np.arctan2(w[0][1], w[0][0])
  #     angle = 180. * angle / np.pi  # convert to degrees
  #     v = 2. * np.sqrt(2.) * np.sqrt(v)
  #     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
  #     ell.set_clip_box(splot.bbox)
  #     ell.set_alpha(.5)
  #     splot.add_artist(ell)

  # plt.xticks(())
  # plt.yticks(())
  # plt.title('Selected GMM: full model, 2 components')
  # plt.subplots_adjust(hspace=.35, bottom=.02)
  # plt.savefig(title+'EMcomponents.png')



# Breast Cancer
data = pd.read_csv('breast.csv')
b_labels = data['Class']
data = data.values
plt.figure()
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,20), title="BC Distortion Score Elbow for KMeans")
visualizer.fit(data)        # Fit the data to the visualizer
visualizer.show('bcKmeansElbow.png')
n_samples, breast_n_features = data.shape
n_digits = 2
breast_data = scale(data)

timings = defaultdict(dict)

run_cluster(breast_data, b_labels, n_samples, breast_n_features, n_digits, "Breast Cancer", True)


#Digits
data, d_labels = load_digits(return_X_y=True)
model = KMeans()

plt.figure()
visualizer = KElbowVisualizer(model, k=(4,20), title="Digits Distortion Score Elbow for KMeans")
visualizer.fit(data)        # Fit the data to the visualizer
# visualizer.show()        # Finalize and render the figur
visualizer.show('digitsKmeansElbow.png')
digits_data = scale(data)
n_samples, digits_n_features = data.shape
n_digits = len(np.unique(d_labels))
run_cluster(digits_data, d_labels, n_samples, digits_n_features, n_digits, "Digits", True)

b_pca = PCA(n_components=7)
b_pca2_results = b_pca.fit_transform(breast_data)
run_cluster(b_pca2_results, b_labels, n_samples, breast_n_features, n_digits, "Breast Cancer PCA 7")
clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
timings['bc']['PCA'] = 0
start = clock()
util.plot_learning_curve(clf, "B Cancer PCA ANN LC", b_pca2_results, b_labels, cv=5, n_jobs=-1)
timings['bc']['PCA'] += clock() - start


b_ica = FastICA(n_components=7)
temp = b_ica.fit_transform(breast_data)
run_cluster(temp, b_labels, n_samples, breast_n_features, n_digits, "Breast Cancer ICA 7")
clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
timings['bc']['ICA'] = 0
start = clock()
util.plot_learning_curve(clf, "B Cancer ICA ANN LC", temp, b_labels, cv=5, n_jobs=-1)
timings['bc']['ICA'] += clock() - start



RP = random_projection.GaussianRandomProjection(n_components=7)
rp2_results = RP.fit_transform(breast_data)
run_cluster(rp2_results, b_labels, n_samples, breast_n_features, n_digits, "Breast Cancer RP 7")
clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
timings['bc']['RP'] = 0
start = clock()
util.plot_learning_curve(clf, "B Cancer RP ANN LC", rp2_results, b_labels, cv=5, n_jobs=-1)
timings['bc']['RP'] += clock() - start

b_lda = FactorAnalysis(n_components=7)
b_lda2_results = b_lda.fit_transform(breast_data, b_labels)
run_cluster(b_lda2_results, b_labels, n_samples, breast_n_features, n_digits, "Breast Cancer FA 7")
clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
timings['bc']['FA'] = 0
start = clock()
util.plot_learning_curve(clf, "B Cancer FA ANN LC", b_lda2_results, b_labels, cv=5, n_jobs=-1)
timings['bc']['FA'] += clock() - start


# b_lda = ld(n_components=7)
# b_lda2_results = b_lda.fit_transform(breast_data, b_labels)
# run_cluster(b_lda2_results, b_labels, n_samples, breast_n_features, n_digits, "Breast Cancer LDA 7")
# clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
# timings['bc']['LDA'] = 0
# start = clock()
# util.plot_learning_curve(clf, "B Cancer LDA ANN LC", b_lda2_results, b_labels, cv=5, n_jobs=-1)
# timings['bc']['LDA'] += clock() - start


d_pca = PCA(n_components=30)
d_pca2_results = d_pca.fit_transform(digits_data)
run_cluster(d_pca2_results, d_labels, n_samples, digits_n_features, n_digits, "Digits PCA 30")
clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
timings['digits']['PCA'] = 0
start = clock()
util.plot_learning_curve(clf, "Digits PCA ANN LC", d_pca2_results, d_labels, cv=5, n_jobs=-1)
timings['digits']['PCA'] += clock() - start


d_ica = FastICA(n_components=30)
temp = d_ica.fit_transform(digits_data)
run_cluster(temp, d_labels, n_samples, digits_n_features, n_digits, "Digits ICA 30")
clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
timings['digits']['ICA'] = 0
start = clock()
util.plot_learning_curve(clf, "Digits ICA ANN LC", temp, d_labels, cv=5, n_jobs=-1)
timings['digits']['ICA'] += clock() - start


RP = random_projection.GaussianRandomProjection(n_components=30)
rp2_results = RP.fit_transform(digits_data)
run_cluster(rp2_results, d_labels, n_samples, digits_n_features, n_digits, "Digits RP 30")
clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
timings['digits']['RP'] = 0
start = clock()
util.plot_learning_curve(clf, "Digits RP ANN LC", rp2_results, d_labels, cv=5, n_jobs=-1)
timings['digits']['RP'] += clock() - start

d_lda = FactorAnalysis(n_components=30)
d_lda2_results = d_lda.fit_transform(digits_data, d_labels)
run_cluster(d_lda2_results, d_labels, n_samples, digits_n_features, n_digits, "Digits FA 30")
clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
timings['digits']['FA'] = 0
start = clock()
util.plot_learning_curve(clf, "Digits FA ANN LC", d_lda2_results, d_labels, cv=5, n_jobs=-1)
timings['digits']['FA'] += clock() - start


# d_lda = ld(n_components=30)
# d_lda2_results = d_lda.fit_transform(digits_data, d_labels)
# run_cluster(d_lda2_results, d_labels, n_samples, digits_n_features, n_digits, "Digits LDA 30")
# clf = MLPClassifier(solver='adam', max_iter = 1000, hidden_layer_sizes=(100,5))
# timings['digits']['LDA'] = 0
# start = clock()
# util.plot_learning_curve(clf, "Digits LDA ANN LC", d_lda2_results, d_labels, cv=5, n_jobs=-1)
# timings['digits']['LDA'] += clock() - start


timings = pd.DataFrame(timings)
timings.to_csv('./clusterANN_timing.csv')

#kmeans relies on isotropy, whiten=True projects data to singular space and scales each component to unit variance, which is important for Kmeans
# print("data",data)
# pca = PCA(n_components=n_digits).fit(data)
# print("pca",pca.components_)
# bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#                 name="PCA-based",
#                 data=data)

# transformer = FastICA(n_components=n_digits)
# X_transformed = transformer.fit_transform(X)


# transformer = random_projection.GaussianRandomProjection()
# X_new = transformer.fit_transform(X)

# from sklearn.decomposition import NMF
# model = NMF(n_components=2, init='random', random_state=0)
# W = model.fit_transform(X)
# H = model.components_

#"borrowed" from https://www.kaggle.com/ericlikedata/reconstruct-error-of-pca


# start=2
# d_variance_record=[]
# b_variance_record=[]
# ica_error = []
# rp_error = []
# lda_error = []

# # pca = PCA(n_components=20)
# # pca2_results = pca.fit_transform(data)
# # print(pca.explained_variance_ratio_.sum())
# # ica = FastICA(n_components=20)
# # ica2_results = ica.fit_transform(data)
# # print(ica.explained_variance_ratio_.sum())


# for i in range(start,digits_n_features):
#     d_pca = PCA(n_components=i)
#     d_pca2_results = d_pca.fit_transform(digits_data)
#     d_variance_record.append(d_pca.explained_variance_ratio_.sum())


# for i in range(start,breast_n_features):
#     b_pca = PCA(n_components=i)
#     b_pca2_results = b_pca.fit_transform(breast_data)
#     b_variance_record.append(b_pca.explained_variance_ratio_.sum())

# plt.clf()
# plt.figure()
# plt.title("PCA Explained Variance by Component #")
# plt.plot(d_variance_record,'r-', alpha=0.7, label="Digits")
# plt.plot(b_variance_record,'b-', alpha=0.7, label="Breast")

# plt.xticks(range(len(d_variance_record)), range(start,digits_n_features), rotation='vertical')
# plt.xlim([-1, len(d_variance_record)])
# plt.legend(loc="best")
# plt.savefig('PCA.png')



#     pca2_proj_back=pca.inverse_transform(pca2_results)
#     total_loss=LA.norm((data-pca2_proj_back),None)
#     error_record.append(total_loss)

#     ica = FastICA(n_components=i)
#     ica2_results = ica.fit_transform(data)
#     ica2_proj_back=ica.inverse_transform(ica2_results)
#     total_loss_ica=LA.norm((data-ica2_proj_back),None)
#     ica_error.append(total_loss_ica)

#     RP = random_projection.GaussianRandomProjection(n_components=i)
#     rp2_results = RP.fit(data)
#     W = rp2_results.components_
#     p = LA.pinv(W)
#     reconstructed = ((p@W)@(data.T)).T
#     errors = np.square(data - reconstructed)
#     # rp2_proj_back=RP.inverse_transform(rp2_results)
#     # total_loss_rp=LA.norm((data-rp2_proj_back),None)
#     rp_error.append(np.nanmean(errors))

    # lda = ld(n_components=i)
    # lda2_results = lda.fit_transform(data, y=None)
    # lda2_proj_back=lda.inverse_transform(lda2_results)
    # total_loss_lda=LA.norm((data-lda2_proj_back),None)
    # lda_error.append(total_loss_lda)








  # print(82 * '_')

  # #############################################################################
  # Visualize the results on PCA-reduced data
  # reduced_data = PCA(n_components=2).fit_transform(data)
  # kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
  # kmeans.fit(reduced_data)

  # # Step size of the mesh. Decrease to increase the quality of the VQ.
  # h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

  # # Plot the decision boundary. For that, we will assign a color to each
  # x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
  # y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
  # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # # Obtain labels for each point in mesh. Use last trained model.
  # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

  # # Put the result into a color plot
  # Z = Z.reshape(xx.shape)
  # plt.figure(1)
  # plt.clf()
  # plt.imshow(Z, interpolation='nearest',
  #           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
  #           cmap=plt.cm.Paired,
  #           aspect='auto', origin='lower')

  # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
  # # Plot the centroids as a white X
  # centroids = kmeans.cluster_centers_
  # plt.scatter(centroids[:, 0], centroids[:, 1],
  #             marker='x', s=169, linewidths=3,
  #             color='w', zorder=10)
  # plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
  #           'Centroids are marked with white cross')
  # plt.xlim(x_min, x_max)
  # plt.ylim(y_min, y_max)
  # plt.xticks(())
  # plt.yticks(())
  # plt.show()
