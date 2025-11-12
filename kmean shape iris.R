
data(iris)
X <- as.matrix(iris[, 1:4])
scaled_data <- scale(X)

set.seed(42)
pca_result <- prcomp(scaled_data, center = TRUE, scale. = FALSE)
pca_data <- as.data.frame(pca_result$x[, 1:2])

# Apply K-Means to the 2-dimensional PCA data, expecting 3 clusters
kmeans_pca_result <- kmeans(pca_data, centers = 3, nstart = 20)

# Add the cluster results back to the PCA data frame
pca_data$Cluster <- factor(kmeans_pca_result$cluster)
pca_data$Species <- iris$Species # Add true labels for comparison

library(ggplot2)

# Plotting K-Means Clusters against the first two Principal Components
ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3) +
  labs(
    title = "K-Means Clusters on PCA-Reduced Iris Data",
    subtitle = "Color shows K-Means Clusters (unsupervised)",
    x = paste0("Principal Component 1 (", round(summary(pca_result)$importance[2, 1] * 100, 1), "%)"),
    y = paste0("Principal Component 2 (", round(summary(pca_result)$importance[2, 2] * 100, 1), "%)")
  ) +
  theme_minimal()

plot_species <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Species, shape = Species)) +
  # Using geom_point to apply both color and shape mapping
  geom_point(size = 3) +
  labs(
    title = "Plot 2: True Species Values on PCA Data (Supervised)",
    subtitle = "Color and Shape show the actual Species (setosa, versicolor, virginica)",
    x = "Principal Component 1",
    y = "Principal Component 2"
  ) +
  # Manually specifying shapes for clarity (e.g., Circle, Square, Triangle)
  scale_shape_manual(values = c(16, 17, 15)) + 
  theme_minimal()

print(plot_species)