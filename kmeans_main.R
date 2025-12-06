source("kmeans_helpers.R")

# Full K-Means Algorithm
kmeans_iterations <- function(data, k, max_iter = 100, tol = 1e-4,
                              use_kmeanspp = FALSE, visualize_init = FALSE) {
  #set.seed(42)
  
  # Initialization step
  if (use_kmeanspp) {
    centroids <- kmeans_pp(data, k, visualize = visualize_init)
  } else {
    n <- nrow(data)
    init_idx <- sample(1:n, k)
    centroids <- data[init_idx, ]
    
    # FIX: ensure column names match for PCA prediction
    colnames(centroids) <- colnames(data)
    
  }
  
  # K-Means Iterations
  history <- list()
  for (i in 1:max_iter) {
    clusters <- assignClusters(data, centroids)
    history[[i]] <- list(iter = i, centroids = centroids, clusters = clusters)
    
    centroids_new <- updateCentroids(data, clusters, k)
    shift <- sum((centroids - centroids_new)^2)
    centroids <- centroids_new
    
    if (shift < tol) {
      cat("Converged at iteration", i, "\n")
      break
    }
  }
  
  return(history)
}
