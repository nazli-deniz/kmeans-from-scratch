
# Load scratch implementation

source("kmeans_helpers.R")
source("kmeans_main.R")

# Load required libraries

library(ggplot2)
library(gridExtra)
library(clue)  # For Hungarian algorithm (solve_LSAP)

# Load Heart dataset

heart_data <- read.csv("heart.csv")
colnames(heart_data) <- c(
  "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
  "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
)

# Scale features

scaled_data <- scale(heart_data)
y_true <- heart_data$num  # True labels
k_chosen <- 2

# Elbow Method

wss <- sapply(1:10, function(k){
  kmeans(scaled_data, k, nstart = 25)$tot.withinss
})

elbow_plot <- ggplot(data.frame(k = 1:10, WSS = wss), aes(x = k, y = WSS)) +
  geom_point() + geom_line() +
  labs(title = "Elbow Method for Heart Dataset",
       x = "Number of clusters k",
       y = "Total Within-Cluster Sum of Squares") +
  theme_minimal()

# Built-in K-means

km_builtin <- kmeans(scaled_data, centers = k_chosen, nstart = 25)
builtin_clusters <- km_builtin$cluster

# Scratch K-means (Random Init)
set.seed(100)
scratch_random_hist <- kmeans_iterations(
  scaled_data, k = k_chosen, max_iter = 100, use_kmeanspp = TRUE
)

final_random <- tail(scratch_random_hist, 1)[[1]]
scratch_random_clusters <- final_random$clusters
scratch_random_centroids <- final_random$centroids

# Scratch K-means++

scratch_pp_hist <- kmeans_iterations(
  scaled_data, k = k_chosen, max_iter = 100, use_kmeanspp = FALSE
)

final_pp <- tail(scratch_pp_hist, 1)[[1]]
scratch_pp_clusters <- final_pp$clusters
scratch_pp_centroids <- final_pp$centroids

# Cluster-to-label mapping (Hungarian algorithm)

map_clusters_to_labels <- function(clusters, labels) {
  clusters <- as.numeric(clusters)
  labels_num <- as.numeric(factor(labels))
  k <- length(unique(clusters))
  tab <- table(clusters, labels_num)
  assignment <- solve_LSAP(tab, maximum = TRUE)
  new_clusters <- clusters
  for (i in 1:k) new_clusters[clusters == i] <- assignment[i]
  factor(new_clusters, labels = levels(factor(labels)))
}

scratch_matched <- map_clusters_to_labels(scratch_random_clusters, y_true)
pp_matched      <- map_clusters_to_labels(scratch_pp_clusters, y_true)
builtin_matched <- map_clusters_to_labels(builtin_clusters, y_true)

# Accuracies

acc_scratch <- mean(scratch_matched == y_true)
acc_pp      <- mean(pp_matched == y_true)
acc_builtin <- mean(builtin_matched == y_true)

# PCA

pca_res <- prcomp(scaled_data, center = TRUE, scale. = TRUE)
pca_data <- data.frame(
  pca_res$x[, 1:2],
  ScratchCluster = factor(scratch_random_clusters),
  PPCluster = factor(scratch_pp_clusters),
  BuiltinCluster = factor(builtin_clusters),
  ScratchMatched = scratch_matched,
  PPMatched = pp_matched,
  BuiltinMatched = builtin_matched,
  TrueLabel = factor(y_true)
)

# Project centroids

pp_centroids_pca <- predict(pca_res, newdata = scratch_pp_centroids)
rand_centroids_pca <- predict(pca_res, newdata = scratch_random_centroids)

# Scratch PCA plot

p_scratch <- ggplot(pca_data, aes(x = PC1, y = PC2, color = ScratchMatched, shape = TrueLabel)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = paste0("Scratch K-Means (Random Init) | Accuracy: ", round(acc_scratch * 100, 1), "%"),
       x = "PC1", y = "PC2") +
  theme_minimal()

# Scratch K-means++ PCA plot

p_pp <- ggplot(pca_data, aes(x = PC1, y = PC2, color = PPMatched, shape = TrueLabel)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = paste0("Scratch K-Means++ | Accuracy: ", round(acc_pp * 100, 1), "%"),
       x = "PC1", y = "PC2") +
  theme_minimal()

# Built-in PCA plot

p_builtin <- ggplot(pca_data, aes(x = PC1, y = PC2, color = BuiltinMatched, shape = TrueLabel)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = paste0("Built-in K-Means | Accuracy: ", round(acc_builtin * 100, 1), "%"),
       x = "PC1", y = "PC2") +
  theme_minimal()

# Centroids overlay for K-means++

pp_centroids_df <- data.frame(pp_centroids_pca[,1:2])
p_centroids <- ggplot(pca_data, aes(PC1, PC2, color = PPMatched)) +
  geom_point(alpha = 0.4) +
  geom_point(data = pp_centroids_df, aes(PC1, PC2), shape = 8, size = 4, color = "black") +
  labs(title = paste0("K-means++ Centroids (Accuracy: ", round(acc_pp * 100, 1), "%)")) +
  theme_minimal()

# Confusion tables

cat("Scratch K-means vs true labels:\n")
print(table(Predicted = scratch_matched, Actual = y_true))

cat("\nScratch K-means++ vs true labels:\n")
print(table(Predicted = pp_matched, Actual = y_true))

cat("\nBuilt-in K-means vs true labels:\n")
print(table(Predicted = builtin_matched, Actual = y_true))

# Display PCA plots and elbow plot

grid.arrange(p_scratch, p_pp, p_builtin, ncol = 2)
print(elbow_plot)
