# Helper Functions

# assign each point to the closest centroid
assignClusters <- function(data, centroids) {
  cluster_assignments <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    point <- data[i, ]
    distances <- apply(centroids, 1, function(c) sqrt(sum((point - c)^2)))
    cluster_assignments[i] <- which.min(distances)
  }
  return(cluster_assignments)
}

# update centroids points in each cluster
updateCentroids <- function(data, clusters, k) {
  centroids <- matrix(0, nrow = k, ncol = ncol(data))
  for (i in 1:k) {
    cluster_points <- data[clusters == i, , drop = FALSE]
    centroids[i, ] <- colMeans(cluster_points)
  }
  colnames(centroids) <- colnames(data)
  return(centroids)
}

# K-Means++ Initialization
kmeans_pp <- function(data, k, visualize = FALSE) {
  n <- nrow(data)
  centroids <- matrix(NA, nrow = k, ncol = ncol(data))
  colnames(centroids) <- colnames(data)
  
  # Pick first centroid randomly
  centroids[1, ] <- data[sample(1:n, 1), ]
  
  # Select remaining centroids based on distance probabilities
  #This ensures that points far from existing centroids are more likely to become new centroids.
  for (i in 2:k) {
    dist_sq <- apply(data, 1, function(x)
      min(colSums((t(centroids[1:(i - 1), ]) - x)^2)))
    probs <- dist_sq / sum(dist_sq)
    cum_probs <- cumsum(probs)
    r <- runif(1)
    next_idx <- which(cum_probs >= r)[1]
    centroids[i, ] <- data[next_idx, ]
  }
  
  # Visualization mode
  if (visualize) {
    library(ggplot2)
    pca_result <- prcomp(data)
    pca_data <- data.frame(pca_result$x[, 1:2])
    colnames(pca_data) <- c("PC1", "PC2")
    
    centroids_centered <- sweep(centroids, 2, pca_result$center, "-")  # subtract mean from each column
    centroids_pca <- centroids_centered %*% pca_result$rotation[, 1:2]  # multiply by rotation for first 2 PCs
    centroids_pca_df <- data.frame(centroids_pca)
    colnames(centroids_pca_df) <- c("PC1", "PC2")
    
    p<-ggplot(pca_data, aes(PC1, PC2)) +
      geom_point(alpha = 0.4, color = "gray60") +
      geom_point(data = centroids_pca_df, aes(x = PC1, y = PC2),
                 color = "red", size = 4, shape = 8) +
      ggtitle("K-Means++ Initialization (Centroids in PCA Space)") +
      theme_minimal(base_size = 14)
    print(p)
  }
  
  return(centroids)
}