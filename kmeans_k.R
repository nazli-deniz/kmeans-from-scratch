#===============================================================
#   ELBOW METHOD + SILHOUETTE METHOD for IRIS using kmeans()
#===============================================================

data(iris)
X <- iris[, 1:4]
scaled_data <- scale(X)

library(cluster)
library(ggplot2)

#===============================================================
# 1) ELBOW METHOD
#===============================================================

k_range <- 1:10
wss <- numeric(length(k_range))

for (k in k_range) {
  km <- kmeans(scaled_data, centers = k, nstart = 25)
  wss[k] <- km$tot.withinss
}

# Plot Elbow Method
plot(k_range, wss, type = "b", pch = 19,
     xlab = "K", ylab = "Total Within Sum of Squares",
     main = "Elbow Method on Iris Dataset")

#===============================================================
# 2) SILHOUETTE METHOD
#===============================================================

k_range <- 2:10
silhouette_scores <- numeric(length(k_range))

for (i in seq_along(k_range)) {
  k <- k_range[i]
  km <- kmeans(scaled_data, centers = k, nstart = 25)
  sil <- silhouette(km$cluster, dist(scaled_data))
  silhouette_scores[i] <- summary(sil)$avg.width
}

# Plot Silhouette Analysis
plot(k_range, silhouette_scores, type = "b", pch = 19,
     xlab = "K", ylab = "Average Silhouette Score",
     main = "Silhouette Method on Iris Dataset")

optimal_k <- k_range[which.max(silhouette_scores)]
abline(v = optimal_k, col = "red", lty = 2)

cat("Optimal K based on Silhouette:", optimal_k, "\n")
