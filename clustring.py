import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


with open("D:\\new_ir\\DocumentVector.pkl", "rb") as file:
    tfidf_matrix = pickle.load(file)
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(tfidf_matrix)

# تخفيض الأبعاد باستخدام SVD
svd = TruncatedSVD(n_components=2)
X_2d = svd.fit_transform(tfidf_matrix)

plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for cluster in range(5):
    plt.scatter(X_2d[clusters == cluster, 0], X_2d[clusters == cluster, 1],
                label=f'Cluster {cluster}', alpha=0.5, s=50)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters , cmap='rainbow', alpha=0.5)
plt.title("Clusters Visualization")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(5)])
plt.show()




####################################
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

def create_clusters(doc_vector, num_clusters=10):
    model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
    model.fit(doc_vector)
    return model

def visualize_clusters(doc_vector, model):
    # Reduce the dimensionality of the data to 2 dimensions using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(doc_vector)
    
    # Get the cluster labels
    labels = model.labels_
    
    # Create a scatter plot of the reduced data
    plt.figure(figsize=(10, 7))
    
    # Plot each cluster with a different color
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = reduced_data[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], label=f'Cluster {k}')
    
    plt.title('PCA of Document Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

# Load your document vectors from the pickle file
with open("D:\\new_ir\\DocumentVector.pkl", "rb") as file:
    tfidf_matrix = pickle.load(file)

# Convert to csr_matrix if not already in that format
if not isinstance(tfidf_matrix, csr_matrix):
    tfidf_matrix = csr_matrix(tfidf_matrix)

# Use SelectKBest to select the top k features
k = 5000  # You can adjust this number based on your memory capacity
selector = SelectKBest(chi2, k=k)
reduced_tfidf_matrix = selector.fit_transform(tfidf_matrix, np.zeros(tfidf_matrix.shape[0]))

# Create the clusters
model = create_clusters(reduced_tfidf_matrix.toarray())
print(model)

# Visualize the clusters
visualize_clusters(reduced_tfidf_matrix.toarray(), model)