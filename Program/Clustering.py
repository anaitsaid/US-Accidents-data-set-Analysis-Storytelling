from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from yellowbrick.cluster import KElbowVisualizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
from statistics import mode
import seaborn as se
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# Load data ------------------------------------------------------------------------------------------------------------
df = pd.read_csv('SampleNormalized&Encoded.csv', index_col=[0])
X1 = df.iloc[:, 0]
X1 = pd.concat([X1, df.select_dtypes(include=[float])], axis=1)
X = df[['Severity', 'Precipitation(in)', 'Distance(mi)', 'Visibility(mi)']]

# Data Dimensionality Reduction for Visualization ----------------------------------------------------------------------
reduced2 = PCA(n_components=2, random_state=42).fit_transform(df)  # Dimensionality reduction for data in 2D
reduced3 = PCA(n_components=3, random_state=42).fit_transform(df)  # Dimensionality reduction for data in 3D
PCA2D = pd.DataFrame(reduced2, columns=['pca1', 'pca2'])
PCA3D = pd.DataFrame(reduced3, columns=['pca1', 'pca2', 'pca3'])


# ----------------------------------------------------------------------------------------------------------------------
# Kmeans Clustering ----------------------------------------------------------------------------------------------------

# Feature Selection ----------------------------------------------------------------------------------------------------
def progressiveFeatureSelection(DF, n_clusters=4, max_features=4,):
    # very basic implementation of an algorithm for feature selection (unsupervised clustering); inspired by this post:
    # https://datascience.stackexchange.com/questions/67040/how-to-do-feature-selection-for-clustering-and-implement-it-in-python
    feature_list = list(DF.columns)
    selected_features = list()
    # select starting feature
    initial_feature = ""
    high_score = 0
    for feature in feature_list:
        Kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data_ = DF[feature]
        labels = Kmeans.fit_predict(data_.to_frame())
        score_ = silhouette_score(data_.to_frame(), labels)
        print("Proposed new feature {} with score {}". format(feature, score_))
        if score_ >= high_score:
            initial_feature = feature
            high_score = score_
    print("The initial feature is {} with a silhouette score of {}.".format(initial_feature, high_score))
    feature_list.remove(initial_feature)
    selected_features.append(initial_feature)
    for _ in range(max_features-1):
        high_score = 0
        selected_feature = ""
        print("Starting selection {}...".format(_))
        for feature in feature_list:
            selection_ = selected_features.copy()
            selection_.append(feature)
            Kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data_ = DF[selection_]
            labels = Kmeans.fit_predict(data_)
            score_ = silhouette_score(data_, labels)
            print("Proposed new feature {} with score {}". format(feature, score_))
            if score_ > high_score:
                selected_feature = feature
                high_score = score_
        selected_features.append(selected_feature)
        feature_list.remove(selected_feature)
        print("Selected new feature {} with score {}". format(selected_feature, high_score))
    return selected_features


# SSE Plot / Elbow Method for finding optimal number of clusters for K-Means Clustering --------------------------------
km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2, 10))
visualizer.fit(X)  # Fit the data to the visualizer
visualizer.show()  # Finalize and render the figure

# Manual Elbow Method for finding optimal number of clusters for K-Means Clustering ------------------------------------
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
fig = go.Figure(data=go.Scatter(x=np.arange(1, 11), y=inertia))
fig.update_layout(title="Inertia vs Cluster Number", xaxis=dict(range=[0, 11], title="Cluster Number"),
                  yaxis={'title': 'Inertia'}, annotations=[
        dict(
            x=4,
            y=inertia[3],
            xref="x",
            yref="y",
            text="Elbow",
            showarrow=True,
            arrowhead=7,
            ax=20,
            ay=-40
        )
    ])
fig.write_html('K-MeansElbowPlot.html')

# Kmeans Clustering ----------------------------------------------------------------------------------------------------
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
df['clusters1'] = kmeans.fit_predict(X)
# 2D & 3D scatter plot
fig = px.scatter(PCA2D, x="pca1", y="pca2", color=df["clusters1"], title="K-means Clustering | Clusters:4 | PCA: 2D")
fig.write_html('K-MeansClustering_2d.html')
fig = px.scatter_3d(PCA3D, x="pca1", y="pca2", z="pca3", color=df["clusters1"], title="K-means Clustering | Clusters:4 | PCA: 3D")
fig.write_html('K-MeansClustering_3d.html')

# Clustering Metrics ---------------------------------------------------------------------------------------------------
score1 = silhouette_score(X, kmeans.labels_, metric='euclidean')
print('\nK-means Clustering Silhouette Score: %.3f' % score1)
score2 = calinski_harabasz_score(X, kmeans.labels_)
print('K-means Clustering Calinski Harabasz Score: %.3f' % score2)
score3 = davies_bouldin_score(X, kmeans.labels_)
print('K-means Clustering Davies Bouldin Score: %.3f\n' % score3)
# We assume that K-Means Clustering is the best ------------------------------------------------------------------------
Best = dict()
Best["S"] = ("K-Means", score1)
Best["C"] = ("K-Means", score2)
Best["D"] = ("K-Means", score3)
Worst = dict()
Worst["S"] = ("K-Means", score1)
Worst["C"] = ("K-Means", score2)
Worst["D"] = ("K-Means", score3)

# ----------------------------------------------------------------------------------------------------------------------
# Spectral Clustering --------------------------------------------------------------------------------------------------
spec = SpectralClustering(n_clusters=4, random_state=0, affinity='nearest_neighbors')
df['clusters2'] = spec.fit_predict(X1)
# 2D & 3D scatter plot
fig = px.scatter(PCA2D, x="pca1", y="pca2", color=df["clusters2"], title="Spectral Clustering | Clusters:4 | PCA: 2D")
fig.write_html('SpectralClustering_2d.html')
fig = px.scatter_3d(PCA3D, x="pca1", y="pca2", z="pca3", color=df["clusters2"], title="Spectral Clustering | Clusters:4 | PCA: 3D")
fig.write_html('SpectralClustering_3d.html')

# Clustering Metrics ---------------------------------------------------------------------------------------------------
score1 = silhouette_score(X, spec.labels_, metric='euclidean')
print('Spectral Clustering Silhouette Score: %.3f' % score1)
score2 = calinski_harabasz_score(X, spec.labels_)
print('Spectral Clustering Calinski Harabasz Score: %.3f' % score2)
score3 = davies_bouldin_score(X, spec.labels_)
print('Spectral Clustering Davies Bouldin Score: %.3f\n' % score3)

# Checking if Spectral Clustering is better ----------------------------------------------------------------------------
if score1 > Best["S"][1]:
    Best["S"] = ("Spectral", score1)
if score2 > Best["C"][1]:
    Best["C"] = ("Spectral", score2)
if score2 < Best["D"][1]:
    Best["D"] = ("Spectral", score3)
# OR Worse -------------------------------------------------------------------------------------------------------------
if score1 < Worst["S"][1]:
    Worst["S"] = ("Spectral", score1)
if score2 < Worst["C"][1]:
    Worst["C"] = ("Spectral", score2)
if score2 > Worst["D"][1]:
    Worst["D"] = ("Spectral", score3)

# ----------------------------------------------------------------------------------------------------------------------
# DBSCAN Clustering ----------------------------------------------------------------------------------------------------

# Choosing the right epsilon parameter ---------------------------------------------------------------------------------
def findOptimalEps(n_neighbors, data):
    # function to find optimal eps distance when using DBSCAN; based on this article:
    # https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()


# DBSCAN Clustering ----------------------------------------------------------------------------------------------------
dbscan = DBSCAN(eps=0.2, metric="euclidean")
df['clusters3'] = dbscan.fit_predict(X)
# 2D & 3D scatter plot
fig = px.scatter(PCA2D, x="pca1", y="pca2", color=df["clusters3"], title="DBSCAN Clustering | PCA: 2D")
fig.write_html('DBSCANClustering_2d.html')
fig = px.scatter_3d(PCA3D, x="pca1", y="pca2", z="pca3", color=df["clusters3"], title="DBSCAN Clustering | PCA: 3D")
fig.write_html('DBSCANClustering_3d.html')

# Clustering Metrics ---------------------------------------------------------------------------------------------------
score1 = silhouette_score(X, dbscan.labels_, metric='euclidean')
print('DBSCAN Clustering Silhouette Score: %.3f' % score1)
score2 = calinski_harabasz_score(X, dbscan.labels_)
print('DBSCAN Clustering Calinski Harabasz Score: %.3f' % score2)
score3 = davies_bouldin_score(X, dbscan.labels_)
print('DBSCAN Clustering Davies Bouldin Score: %.3f\n' % score3)

# Checking if Spectral Clustering is better ----------------------------------------------------------------------------
if score1 > Best["S"][1]:
    Best["S"] = ("DBSCAN", score1)
if score2 > Best["C"][1]:
    Best["C"] = ("DBSCAN", score2)
if score2 < Best["D"][1]:
    Best["D"] = ("DBSCAN", score3)
# OR Worse -------------------------------------------------------------------------------------------------------------
if score1 < Worst["S"][1]:
    Worst["S"] = ("DBSCAN", score1)
if score2 < Worst["C"][1]:
    Worst["C"] = ("DBSCAN", score2)
if score2 > Worst["D"][1]:
    Worst["D"] = ("DBSCAN", score3)


# Comparison scatter plot ----------------------------------------------------------------------------------------------
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig1.set_size_inches(18.5, 10.5)
fig1.suptitle('K-Means vs Spectral vs DBSCAN Clustering')
ax1.set_title('K-Means')
se.scatterplot(x="pca1", y="pca2", hue=df['clusters1'], data=PCA2D, ax=ax1)
ax2.set_title('Spectral')
se.scatterplot(x="pca1", y="pca2", hue=df['clusters2'], data=PCA2D, ax=ax2)
ax3.set_title('DBSCAN')
se.scatterplot(x="pca1", y="pca2", hue=df['clusters3'], data=PCA2D, ax=ax3)
plt.savefig('ClusteringComparisonFigure.png')

# Best & Worst ---------------------------------------------------------------------------------------------------------
print("Best for Silhouette Score, Calinski Harabasz Score & Davies Bouldin Score respectively\n", Best)
print("Worst for Silhouette Score, Calinski Harabasz Score & Davies Bouldin Score respectively\n", Worst)

Cl = ["K-Means", "Spectral", "DBSCAN"]
print("\nRaking the Clustering methods used from Best to Worst:")
BestList = list(Best.values())
out = [item for t in BestList for item in t]
new = [item for item in out if type(item) == str]
best = mode(new)
print("1. ", best, "Clustering.")
WorstList = list(Worst.values())
out = [item for t in WorstList for item in t]
new = [item for item in out if type(item) == str]
worst = mode(new)
average = "".join([x for x in Cl if x != best and x != worst])
print("2. ", average, "Clustering.")
print("3. ", worst, "Clustering.")

# Disclaimer: the two methods: progressiveFeatureSelection and findOptimalEps are based on advanced researches & studies
# They are written by a specialist, we just used them. They are available through:
# https://programminghistorian.org/en/lessons/clustering-with-scikit-learn-in-python#5-choosing-the-right-amount-of-clusters
