# Load necessary packages
library(mlbench)
library(ggplot2)
library(viridis)
library(pals)
set.seed(42)

make_moons <- function(n_samples = 200, noise = 0.05) {
  n <- n_samples / 2
  
  t1 <- runif(n, 0, pi)
  x1 <- cbind(cos(t1), sin(t1))
  
  t2 <- runif(n, 0, pi)
  x2 <- cbind(1 - cos(t2), 1 - sin(t2) - 0.5)
  
  X <- rbind(x1, x2)
  
  X <- X + matrix(rnorm(n_samples * 2, sd = noise), nrow = n_samples)
  
  df <- as.data.frame(X)
  colnames(df) <- c("Feature1", "Feature2")
  return(df)
}

X <- make_moons(n_samples = 200, noise = 0.05)

# 2. Run K-Means (forcing K=2)
kmeans_result <- kmeans(X, centers = 2, nstart = 25)
X$Cluster <- factor(kmeans_result$cluster)

# 4. Plotting the results
ggplot(X, aes(x = Feature1, y = Feature2, color = Cluster)) +
  geom_point(size = 3) +
  scale_color_viridis_d() +
  labs(
    title = "K-Means Struggling with Non-Spherical Clusters (Moons Data - pals)",
    x = "Feature 1",
    y = "Feature 2",
    color = "Cluster"
  ) +
  theme_minimal(base_size = 15)
