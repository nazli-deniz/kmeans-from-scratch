

data(iris)
X <- as.matrix(iris[, 1:4])
scaled_data <- scale(X)

set.seed(42)
wss <- sapply(1:10, 
              function(k){
                kmeans(scaled_data, k, nstart = 25)$tot.withinss
              })

#plotting elbow
plot(1:10, wss, 
     type="b", 
     xlab="k", 
     ylab="wss",
     main="figure of elbow method for Iris")

k_chosen <- 3
km_result <- kmeans(scaled_data, centers = k_chosen, nstart = 25)

print(km_result)

iris_clustered <- iris
iris_clustered$Cluster <- factor(km_result$cluster)

table(iris_clustered$Cluster)

library(ggplot2)
ggplot(iris_clustered, aes(x = Sepal.Length, y = Petal.Length, color = Cluster)) +
  geom_point(size = 3) +
  labs(color="cluster", x="Sepal.Length", y="Petal.Length", title="K-Means clustring") +
  theme_minimal()
