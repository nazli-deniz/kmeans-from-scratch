source("kmeans_helpers.R")
source("kmeans_main.R")

library(ggplot2)
library(gridExtra)
library(clue)

# Load and preprocess dataset

data(iris)
X <- as.matrix(iris[, 1:4])
scaled_data <- scale(X)
classes <- iris$Species
k_chosen <- length(unique(classes))  # = 3

# Elbow method (built-in K-means)

set.seed(42)
wss <- sapply(1:10, function(k) {
  kmeans(scaled_data, k, nstart = 25)$tot.withinss
})

elbow_plot <- ggplot(data.frame(k = 1:10, WSS = wss), aes(x = k, y = WSS)) +
  geom_point() + geom_line() +
  labs(title = "Elbow Method for Iris (Built-in K-means)",
       x = "Number of clusters k",
       y = "Total Within-Cluster Sum of Squares") +
  theme_minimal()

# Built-in K-means

set.seed(42)
km_builtin <- kmeans(scaled_data, centers = k_chosen, nstart = 25)
builtin_clusters <- km_builtin$cluster

# Scratch K-means (Random Init)

scratch_random_hist <- kmeans_iterations(
  scaled_data,
  k = k_chosen,
  max_iter = 100,
  use_kmeanspp = FALSE
)

final_random <- tail(scratch_random_hist, 1)[[1]]
scratch_random_clusters <- final_random$clusters
scratch_random_centroids <- final_random$centroids

# Scratch K-means++

scratch_pp_hist <- kmeans_iterations(
  scaled_data,
  k = k_chosen,
  max_iter = 100,
  use_kmeanspp = TRUE
)

final_pp <- tail(scratch_pp_hist, 1)[[1]]
scratch_pp_clusters <- final_pp$clusters
scratch_pp_centroids <- final_pp$centroids

# Cluster-to-species mapping (Hungarian algorithm)

map_clusters_to_classes <- function(clusters, classes) {
  clusters <- as.numeric(clusters)
  class_nums <- as.numeric(factor(classes))
  k <- length(unique(clusters))
  
  cont_table <- table(clusters, class_nums)
  assignment <- solve_LSAP(cont_table, maximum = TRUE)
  
  new_clusters <- clusters
  for (i in 1:k) {
    new_clusters[clusters == i] <- assignment[i]
  }
  
  factor(new_clusters, labels = levels(factor(classes)))
}

builtin_matched  <- map_clusters_to_classes(builtin_clusters, classes)
random_matched   <- map_clusters_to_classes(scratch_random_clusters, classes)
pp_matched       <- map_clusters_to_classes(scratch_pp_clusters, classes)

# PCA for visualization

pca_res <- prcomp(scaled_data, center = TRUE, scale. = TRUE)
pca_data <- data.frame(
  pca_res$x[,1:2],
  Builtin = builtin_matched,
  RandomInit = random_matched,
  KmeansPP = pp_matched,
  TrueClass = factor(classes)
)

# Project centroids

pp_centroids_pca <- predict(pca_res, newdata = scratch_pp_centroids)
rand_centroids_pca <- predict(pca_res, newdata = scratch_random_centroids)

# Plotting function

calc_accuracy <- function(pred, true) mean(pred == true)

plot_with_accuracy <- function(pca_data, x_col, y_col, cluster_col, true_col, title, accuracy) {
  ggplot(pca_data, aes_string(x = x_col, y = y_col, color = cluster_col, shape = true_col)) +
    geom_point(size = 3, alpha = 0.8) +
    labs(title = paste0(title, " (Accuracy: ", round(accuracy, 4), ")")) +
    theme_minimal()
}

acc_builtin <- calc_accuracy(builtin_matched, classes)
acc_random  <- calc_accuracy(random_matched, classes)
acc_pp      <- calc_accuracy(pp_matched, classes)

p_builtin <- plot_with_accuracy(pca_data, "PC1", "PC2", "Builtin", "TrueClass", "Built-in K-means", acc_builtin)
p_random  <- plot_with_accuracy(pca_data, "PC1", "PC2", "RandomInit", "TrueClass", "Scratch K-means (Random Init)", acc_random)
p_pp      <- plot_with_accuracy(pca_data, "PC1", "PC2", "KmeansPP", "TrueClass", "Scratch K-means++", acc_pp)

# Centroids overlay for K-means++

pp_centroids_df <- data.frame(pp_centroids_pca[,1:2])
p_centroids <- ggplot(pca_data, aes(PC1, PC2, color = KmeansPP)) +
  geom_point(alpha = 0.4) +
  geom_point(data = pp_centroids_df, aes(PC1, PC2), shape = 8, size = 4, color = "black") +
  labs(title = paste0("K-means++ Centroids (Accuracy: ", round(acc_pp, 4), ")")) +
  theme_minimal()

# Confusion Matrices

cat("Built-in K-means:\n")
print(table(Predicted = builtin_matched, Actual = classes))

cat("\nScratch (Random Init):\n")
print(table(Predicted = random_matched, Actual = classes))

cat("\nScratch (K-means++):\n")
print(table(Predicted = pp_matched, Actual = classes))

# Display plots
grid.arrange(p_builtin, p_random, p_pp, ncol = 2)
print(elbow_plot)
