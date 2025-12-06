# Load scratch implementation
source("kmeans_helpers.R")
source("kmeans_main.R")

# Libraries
library(ggplot2)
library(gridExtra)
library(clue)

# Load wine dataset
library(rattle)
data(wine)

X <- as.matrix(wine[, -1])      # remove class label
scaled_data <- scale(X)
classes <- wine$Type
k_chosen <- length(unique(classes))  # = 3

set.seed(42)
km_builtin <- kmeans(scaled_data, centers = k_chosen, nstart = 25)
builtin_clusters <- km_builtin$cluster

scratch_random_hist <- kmeans_iterations(
  scaled_data,
  k = k_chosen,
  max_iter = 100,
  use_kmeanspp = FALSE
)

final_random <- tail(scratch_random_hist, 1)[[1]]
scratch_random_clusters <- final_random$clusters
scratch_random_centroids <- final_random$centroids

scratch_pp_hist <- kmeans_iterations(
  scaled_data,
  k = k_chosen,
  max_iter = 100,
  use_kmeanspp = TRUE
)

final_pp <- tail(scratch_pp_hist, 1)[[1]]
scratch_pp_clusters <- final_pp$clusters
scratch_pp_centroids <- final_pp$centroids

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
p_builtin <- ggplot(pca_data, aes(PC1, PC2, color = Builtin, shape = TrueClass)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "Built-in K-means (Wine)") +
  theme_minimal()
p_random <- ggplot(pca_data, aes(PC1, PC2, color = RandomInit, shape = TrueClass)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "Scratch K-means (Random Init)") +
  theme_minimal()
p_pp <- ggplot(pca_data, aes(PC1, PC2, color = KmeansPP, shape = TrueClass)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "Scratch K-means++") +
  theme_minimal()



calc_accuracy <- function(pred, true) {
  mean(pred == true)
}

acc_builtin <- calc_accuracy(builtin_matched, classes)
acc_random  <- calc_accuracy(random_matched, classes)
acc_pp      <- calc_accuracy(pp_matched, classes)

plot_with_accuracy <- function(pca_data, x_col, y_col, cluster_col, true_col, title, accuracy) {
  ggplot(pca_data, aes_string(x = x_col, y = y_col, color = cluster_col, shape = true_col)) +
    geom_point(size = 3, alpha = 0.8) +
    labs(title = paste0(title, " (Accuracy: ", round(accuracy, 4), ")")) +
    theme_minimal()
}

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

# Accuracies

cat("\nAccuracies:\n")
cat("Built-in K-means: ", round(acc_builtin, 4), "\n")
cat("Scratch K-means (Random Init): ", round(acc_random, 4), "\n")
cat("Scratch K-means++: ", round(acc_pp, 4), "\n")

# Display Plots
grid.arrange(p_builtin, p_random, p_pp, ncol = 2)
