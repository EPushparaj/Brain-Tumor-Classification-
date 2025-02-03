import numpy as np

def relief_score(X, k=10):

  # Calculate the distance between each sample and all other samples.
  distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

  # Find the k nearest neighbors of each sample.
  nearest_neighbors = np.argsort(distances, axis=1)[:, :k]

  # Calculate the difference between the feature value of each sample and the
  # feature values of its nearest neighbors.
  feature_differences = X[:, np.newaxis] - X[nearest_neighbors]

  # Calculate the Relief score for each feature.
  relief_scores = np.mean(np.abs(feature_differences), axis=1)

  return relief_scores
