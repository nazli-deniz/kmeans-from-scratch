library(ggplot2)
library(gridExtra)
library(mclust)
source("kmeans_main.R")

# Function to calculate Adjusted Rand Index (ARI)
calculate_ari <- function(predicted, true_labels) {
  return(adjustedRandIndex(true_labels, predicted))
}
  
# synthetic dataset: overlapping Gaussian clusters
set.seed(123)
n_per_cluster <- 150
k_true <- 4
  
# Cluster centers arranged in a line with heavy overlap
centers <- matrix(c(0, 0,2, 0,4, 0,6, 0), nrow = 4, byrow = TRUE)
  
# Generate overlapping clusters with moderate spread
data <- do.call(rbind, lapply(1:k_true, function(i) {
cbind(rnorm(n_per_cluster, mean = centers[i, 1], sd = 1.0),
      rnorm(n_per_cluster, mean = centers[i, 2], sd = 1.0))
  }))
  
colnames(data) <- c("X1", "X2")
data <- as.data.frame(data)
  
# Keep true labels for ARI calculation
true_labels <- rep(1:k_true, each = n_per_cluster)
  
# Function to calculate inertia (within-cluster sum of squares)
calculate_inertia <- function(data, clusters, centroids) {
    inertia <- 0
    for (i in 1:nrow(data)) {
      cluster_id <- clusters[i]
      point <- as.numeric(data[i, ])
      centroid <- as.numeric(centroids[cluster_id, ])
      inertia <- inertia + sum((point - centroid)^2)
    }
    return(inertia)
  }
  
  # Run K-means multiple times with random initialization
  set.seed(456)
  n_runs <- 20
  random_results <- list()
  random_inertias <- numeric(n_runs)
  random_accuracies <- numeric(n_runs)
  
  for (run in 1:n_runs) {
    result <- kmeans_iterations(data, k = 4, max_iter = 100, tol = 1e-4, 
                                use_kmeanspp = FALSE, visualize_init = FALSE)
    final_iter <- length(result)
    final_clusters <- result[[final_iter]]$clusters
    final_centroids <- result[[final_iter]]$centroids
    
    inertia <- calculate_inertia(data, final_clusters, final_centroids)
    ari <- calculate_ari(final_clusters, true_labels)
    
    random_inertias[run] <- inertia
    random_accuracies[run] <- ari
    random_results[[run]] <- list(clusters = final_clusters, centroids = final_centroids, 
                                  inertia = inertia, ari = ari)
  }
  
  # Run K-means multiple times with K-means++ initialization
  set.seed(456)
  kmeans_pp_results <- list()
  kmeans_pp_inertias <- numeric(n_runs)
  kmeans_pp_accuracies <- numeric(n_runs)
  
  for (run in 1:n_runs) {
    result <- kmeans_iterations(data, k = 4, max_iter = 100, tol = 1e-4, 
                                use_kmeanspp = TRUE, visualize_init = FALSE)
    final_iter <- length(result)
    final_clusters <- result[[final_iter]]$clusters
    final_centroids <- result[[final_iter]]$centroids
    
    inertia <- calculate_inertia(data, final_clusters, final_centroids)
    ari <- calculate_ari(final_clusters, true_labels)
    
    kmeans_pp_inertias[run] <- inertia
    kmeans_pp_accuracies[run] <- ari
    kmeans_pp_results[[run]] <- list(clusters = final_clusters, centroids = final_centroids, 
                                     inertia = inertia, ari = ari)
  }
  
  # Built-in kmeans() with nstart=20
  set.seed(456)
  builtin_kmeans <- kmeans(data, centers = 4, nstart = 20)
  
  builtin_clusters  <- builtin_kmeans$cluster
  builtin_centroids <- builtin_kmeans$centers
  builtin_inertia   <- builtin_kmeans$tot.withinss
  builtin_ari       <- adjustedRandIndex(builtin_clusters, true_labels)
  
  # Print statistics
  cat("=== RANDOM INITIALIZATION ===\n")
  cat("Mean Inertia:", mean(random_inertias), "\n")
  cat("Std Dev:", sd(random_inertias), "\n")
  cat("Min:", min(random_inertias), "\n")
  cat("Max:", max(random_inertias), "\n\n")
  
  cat("Mean Accuracy:", round(mean(random_accuracies), 4), "\n")
  cat("Std Dev:", round(sd(random_accuracies), 4), "\n")
  cat("Min:", round(min(random_accuracies), 4), "\n")
  cat("Max:", round(max(random_accuracies), 4), "\n\n")
  
  cat("=== K-MEANS++ INITIALIZATION ===\n")
  cat("Mean Inertia:", mean(kmeans_pp_inertias), "\n")
  cat("Std Dev:", sd(kmeans_pp_inertias), "\n")
  cat("Min:", min(kmeans_pp_inertias), "\n")
  cat("Max:", max(kmeans_pp_inertias), "\n\n")
  
  cat("Mean Accuracy:", round(mean(kmeans_pp_accuracies), 4), "\n")
  cat("Std Dev:", round(sd(kmeans_pp_accuracies), 4), "\n")
  cat("Min:", round(min(kmeans_pp_accuracies), 4), "\n")
  cat("Max:", round(max(kmeans_pp_accuracies), 4), "\n\n")
  
  cat("Inertia Improvement (%):", 
      round(((mean(random_inertias) - mean(kmeans_pp_inertias)) / mean(random_inertias) * 100), 2), "\n")
  
  # Plot 1: Inertia comparison
  inertia_df <- data.frame(
    Inertia = c(random_inertias, kmeans_pp_inertias, rep(builtin_inertia, n_runs)),
    Method = rep(c("Random Init", "K-Means++", "Built-in kmeans(nstart=20)"), each = n_runs)
  )
  
  p1 <- ggplot(inertia_df, aes(x = Method, y = Inertia, fill = Method)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.4, size = 2) +
    ggtitle("Inertia Comparison (20 runs)") +
    ylab("Within-Cluster Sum of Squares") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 10, hjust = 1))
  
  # Plot 1b: ARI comparison
  ari_df <- data.frame(
    ARI = c(random_accuracies, kmeans_pp_accuracies, rep(builtin_ari, n_runs)),
    Method = rep(c("Random Init", "K-Means++", "Built-in kmeans(nstart=20)"), each = n_runs)
  )
  
  p1b <- ggplot(ari_df, aes(x = Method, y = ARI, fill = Method)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.4, size = 2) +
    ggtitle("Adjusted Rand Index Comparison (20 runs)") +
    ylab("ARI (ranges from -1 to 1)") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 10, hjust = 1))
  
  # Plot 2: Best result from random initialization → now shows ARI
  best_random_idx <- which.max(random_accuracies)        # changed to max ARI
  best_random <- random_results[[best_random_idx]]
  
  df_random <- data.frame(data, Cluster = as.factor(best_random$clusters))
  p2 <- ggplot(df_random, aes(X1, X2, color = Cluster)) +
    geom_point(alpha = 0.6, size = 2) +
    geom_point(data = as.data.frame(best_random$centroids), 
               aes(x = X1, y = X2), color = "black", size = 5, shape = 8, inherit.aes = FALSE) +
    ggtitle(paste("Best Random Init (ARI =", round(best_random$ari, 4), ")")) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "right")
  
  # Plot 3: Best result from K-means++ → now shows ARI
  best_kmeans_pp_idx <- which.max(kmeans_pp_accuracies)   # changed to max ARI
  best_kmeans_pp <- kmeans_pp_results[[best_kmeans_pp_idx]]
  
  df_kmeans_pp <- data.frame(data, Cluster = as.factor(best_kmeans_pp$clusters))
  p3 <- ggplot(df_kmeans_pp, aes(X1, X2, color = Cluster)) +
    geom_point(alpha = 0.6, size = 2) +
    geom_point(data = as.data.frame(best_kmeans_pp$centroids), 
               aes(x = X1, y = X2), color = "black", size = 5, shape = 8, inherit.aes = FALSE) +
    ggtitle(paste("Best K-Means++ (ARI =", round(best_kmeans_pp$ari, 4), ")")) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "right")
  
  # NEW Plot: Built-in kmeans clustering result
  df_builtin <- data.frame(data, Cluster = as.factor(builtin_clusters))
  p_builtin <- ggplot(df_builtin, aes(X1, X2, color = Cluster)) +
    geom_point(alpha = 0.6, size = 2) +
    geom_point(data = as.data.frame(builtin_centroids), 
               aes(x = X1, y = X2), color = "black", size = 5, shape = 8, inherit.aes = FALSE) +
    ggtitle(paste("Built-in kmeans(nstart=20) (ARI =", round(builtin_ari, 4), ")")) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "right")
  
  # Plot 4: Worst result from random initialization (to show pathological case)
  worst_random_idx <- which.max(random_inertias)
  worst_random <- random_results[[worst_random_idx]]
  
  df_worst_random <- data.frame(data, Cluster = as.factor(worst_random$clusters))
  p4 <- ggplot(df_worst_random, aes(X1, X2, color = Cluster)) +
    geom_point(alpha = 0.6, size = 2) +
    geom_point(data = as.data.frame(worst_random$centroids), 
               aes(x = X1, y = X2), color = "black", size = 5, shape = 8, inherit.aes = FALSE) +
    ggtitle(paste("Worst Random Init (ARI =", round(worst_random$ari, 4), ")")) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "right")
  
  # Combine plots
  grid.arrange(p1, p1b, p2, p3, p_builtin, p4, ncol = 2)
