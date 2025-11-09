# Load dataset
data(iris)

# Drop the species column (label)
iris_data <- iris[, 1:4]

# Scale data
iris_scaled <- scale(iris_data)
head(iris_data)

# Calculating inertia for different K values
inertia <- numeric(10)

for (k in 1:10) {
  km <- kmeans(iris_scaled, centers = k, nstart = 10)
  inertia[k] <- km$tot.withinss
}

# Plot the elbow
plot(1:10, inertia, type = "b", pch = 19,
     xlab = "Number of Clusters (k value)",
     ylab = "Inertia",
     main = "Elbow Method for Choosing k")

# Helper function to calculate euclidean distance
distance <- function(a, b)
{
  sqrt(rowSums((a - b)^2))
}

# Helper function to assign each point to the closest centroid
assignClusters <- function(data, centroids) {
  cluster_assignments <- numeric(nrow(data))  # initialize
  
  for (i in 1:nrow(data)) {
    point <- data[i, ]
    distances <- numeric(nrow(centroids))
    
    for (j in 1:nrow(centroids)) {
      centroid <- centroids[j, ]
      distances[j] <- sqrt(sum((point - centroid)^2))
    }
    
    cluster_assignments[i] <- which.min(distances)
  }
  
  return(cluster_assignments)
}


updateCentroids <- function(data, clusters, k) {
  centroids <- matrix(0, nrow = k, ncol = ncol(data))
  
  for (i in 1:k) {
    cluster_points <- data[clusters == i, , drop = FALSE]
    centroids[i, ] <- colMeans(cluster_points)
  }
  
  return(centroids)
}

kmeansFromScratch <- function(data, k, max_iter = 100, tol = 1e-4) 
{
  n <- nrow(data)
  p <- ncol(data)
  
  # Initialize centroids randomly
  set.seed(123)
  init_idx <- sample(1:n, k)
  centroids <- data[init_idx, ]
  
  # Initialize cluster assignments
  clusters <- rep(0, n)
  
  for (i in 1:max_iter) {
    # Assign points to nearest centroids
    clusters_new <- assignClusters(data, centroids)
    
    # Update centroids
    centroids_new <- updateCentroids(data, clusters_new, k)
    
    # Check for convergence
    centroid_shift <- sum((centroids - centroids_new)^2)
    if (centroid_shift < tol) {
      cat("Converged at iteration", i, "\n")
      break
    }
    
    # Update for next iteration
    centroids <- centroids_new
    clusters <- clusters_new
  }
  
  list(centroids = centroids, clusters = clusters)
}

result <- kmeansFromScratch(iris_scaled, k = 3)
# Compute confusion matrix
conf_mat <- table(result$clusters, iris$Species)

# Automatically find the best label for each cluster
mapping <- apply(conf_mat, 1, function(x) names(which.max(x)))

# Map predicted clusters to species
mapped_clusters <- mapping[as.character(result$clusters)]

# Display comparison table
table(Predicted = mapped_clusters, Actual = iris$Species)
