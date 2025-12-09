library(ggplot2)
library(gridExtra)
library(mclust) # adjustedRandIndex
source("kmeans_main.R")

#Load Heart Dataset
heart <- read.csv("heart.csv")
colnames(heart) <- c("age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal","target")

# Binary target: 0 = no disease, 1 = disease
y_true <- ifelse(heart$target == 0, 1, 2)   # 1=Healthy, 2=Disease
y_true_factor <- factor(y_true, labels = c("Healthy", "Disease"))

# Use only the 13 features
data_matrix <- as.matrix(heart[,1:13])
scaled_data <- scale(data_matrix)

k_chosen <- 2
n_runs <- 20

# Function to compute Inertia
calculate_inertia <- function(data, clusters, centroids) {
  sum(sapply(1:nrow(data), function(i) {
    sum((data[i,] - centroids[clusters[i], ])^2)
  }))
}

# Run Random Init (many times)
set.seed(123)
random_inertias <- numeric(n_runs)
random_aris     <- numeric(n_runs)
random_results  <- list()

for(run in 1:n_runs){
  hist <- kmeans_iterations(scaled_data, k = k_chosen, max_iter = 200,
                            use_kmeanspp = FALSE, visualize_init = FALSE)
  final <- tail(hist,1)[[1]]
  
  random_inertias[run] <- calculate_inertia(scaled_data, final$clusters, final$centroids)
  random_aris[run]     <- adjustedRandIndex(final$clusters, y_true)
  random_results[[run]] <- final
}

# Run K-means++ (many times)
set.seed(123)
pp_inertias <- numeric(n_runs)
pp_aris     <- numeric(n_runs)
pp_results  <- list()

for(run in 1:n_runs){
  hist <- kmeans_iterations(scaled_data, k = k_chosen, max_iter = 200,
                            use_kmeanspp = TRUE, visualize_init = FALSE)
  final <- tail(hist,1)[[1]]
  
  pp_inertias[run] <- calculate_inertia(scaled_data, final$clusters, final$centroids)
  pp_aris[run]     <- adjustedRandIndex(final$clusters, y_true)
  pp_results[[run]] <- final
}

# Built-in kmeans with nstart=20 
set.seed(123)
builtin_kmeans <- kmeans(scaled_data, centers = k_chosen, nstart = 1)

builtin_clusters  <- builtin_kmeans$cluster
builtin_centroids <- builtin_kmeans$centers
builtin_inertia   <- builtin_kmeans$tot.withinss
builtin_ari       <- adjustedRandIndex(builtin_clusters, y_true)

#  Statistics
cat("=== RANDOM INITIALIZATION ===\n")
cat("Mean ARI:    ", round(mean(random_aris), 4), "\n")
cat("Std Dev ARI: ", round(sd(random_aris), 4), "\n")
cat("Mean Inertia:", round(mean(random_inertias), 1), "\n")
cat("Std Dev Inertia:", round(sd(random_inertias), 1), "\n\n")

cat("=== K-MEANS++ INITIALIZATION ===\n")
cat("Mean ARI:    ", round(mean(pp_aris), 4), "\n")
cat("Std Dev ARI: ", round(sd(pp_aris), 4), "\n")
cat("Mean Inertia:", round(mean(pp_inertias), 1), "\n")
cat("Std Dev Inertia:", round(sd(pp_inertias), 1), "\n\n")

cat("=== BUILT-IN kmeans(nstart=20) ===\n")
cat("ARI:     ", round(builtin_ari, 4), "\n")
cat("Inertia: ", round(builtin_inertia, 1), "\n\n")

cat("ARI Improvement (%):", round((mean(pp_aris) - mean(random_aris))/mean(random_aris)*100, 2), "\n")

# Plot 1: Inertia comparison (boxplot) 
inertia_df <- data.frame(
  Inertia = c(random_inertias, pp_inertias, rep(builtin_inertia, n_runs)),
  Method = rep(c("Random Init", "K-Means++", "Built-in kmeans(nstart=20)"), each = n_runs)
)

p1 <- ggplot(inertia_df, aes(x = Method, y = Inertia, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 2) +

  ggtitle(paste("Inertia Comparison (", n_runs, " runs)", sep="")) +
  ylab("Within-Cluster Sum of Squares") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 12, hjust = 1))

#Plot 1b: ARI comparison (boxplot)
ari_df <- data.frame(
  ARI = c(random_aris, pp_aris, rep(builtin_ari, n_runs)),
  Method = rep(c("Random Init", "K-Means++", "Built-in kmeans(nstart=20)"), each = n_runs)
)

p1b <- ggplot(ari_df, aes(x = Method, y = ARI, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 2) +

  ggtitle(paste("Adjusted Rand Index Comparison (", n_runs, " runs)", sep="")) +
  ylab("ARI") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 12, hjust = 1))

# PCA for 2D visualization
pca <- prcomp(scaled_data)

#Plot 2: Best Random Init
best_random_idx <- which.max(random_aris)
best_random <- random_results[[best_random_idx]]
df_random_pca <- data.frame(pca$x[,1:2], Cluster = factor(best_random$clusters))

p2 <- ggplot(df_random_pca, aes(PC1, PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_point(data = as.data.frame(predict(pca, best_random$centroids)),
             aes(PC1, PC2), color = "black", size = 6, shape = 8) +
  ggtitle(paste("Best Random Init (ARI =", round(random_aris[best_random_idx], 4), ")")) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right")

#Plot 3: Best K-Means++ 
best_pp_idx <- which.max(pp_aris)
best_pp <- pp_results[[best_pp_idx]]
df_pp_pca <- data.frame(pca$x[,1:2], Cluster = factor(best_pp$clusters))

p3 <- ggplot(df_pp_pca, aes(PC1, PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_point(data = as.data.frame(predict(pca, best_pp$centroids)),
             aes(PC1, PC2), color = "black", size = 6, shape = 8) +
  ggtitle(paste("Best K-Means++ (ARI =", round(pp_aris[best_pp_idx], 4), ")")) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right")

# Plot 5: Built-in kmeans(nstart=20)
df_builtin_pca <- data.frame(pca$x[,1:2], Cluster = factor(builtin_clusters))

p_builtin <- ggplot(df_builtin_pca, aes(PC1, PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_point(data = as.data.frame(predict(pca, builtin_centroids)),
             aes(PC1, PC2), color = "black", size = 6, shape = 8) +
  ggtitle(paste("Built-in kmeans(nstart=20) (ARI =", round(builtin_ari, 4), ")")) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right")

# Plot 4: Worst Random Init 
worst_random_idx <- which.min(random_aris)
worst_random <- random_results[[worst_random_idx]]
df_worst_pca <- data.frame(pca$x[,1:2], Cluster = factor(worst_random$clusters))

p4 <- ggplot(df_worst_pca, aes(PC1, PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_point(data = as.data.frame(predict(pca, worst_random$centroids)),
             aes(PC1, PC2), color = "black", size = 6, shape = 8) +
  ggtitle(paste("Worst Random Init (ARI =", round(random_aris[worst_random_idx], 4), ")")) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right")

#  Final 6-plot grid 
grid.arrange(p1, p1b, p2, p3, p_builtin, p4, ncol = 2,
             top = "K-Means Initialization Comparison on Heart Disease Dataset (k=2)")

