#loading dataa

library(dplyr)
getwd()
list.files()
heart_data <- read.csv(
  "heart.csv"
)

colnames(heart_data) <- c(
  "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
  "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
)


cat("--- Summary of Missing Values (NA) after loading ---\n")
print(sapply(heart_data, function(x) sum(is.na(x))))


heart_clean <- na.omit(heart_data)

cat("\n--- Structure of Cleaned Data ---\n")
str(heart_clean)

cat("\n--- Final Data Dimensions ---\n")
cat("Original rows:", nrow(heart_data), "\n")
cat("Cleaned rows:", nrow(heart_clean), "\n")
cat("Columns:", ncol(heart_clean), "\n")
summary(heart_clean)

#Pca
heart_features <- heart_clean %>% select(-num) 
scaled_features <- scale(heart_features) 
pca_result <- prcomp(scaled_features, center = TRUE, scale. = FALSE)
pca_data <- as.data.frame(pca_result$x[, 1:2])
variance_explained <- summary(pca_result)$importance[2, 1:2]

#Kmeans
set.seed(42) 

k <- 2
kmeans_pca_result <- kmeans(pca_data, centers = k, nstart = 25)

pca_data$Cluster <- factor(kmeans_pca_result$cluster)
pca_data$True_Label <- factor(heart_clean$num) 

#Performance

print(table(Cluster = pca_data$Cluster, True_Label = pca_data$True_Label))

library(ggplot2)
plot_kmeans <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(
    title = " K-Means",
    x = paste0("PC1 (", round(variance_explained[1] * 100, 1), "%)"),
    y = paste0("PC2 (", round(variance_explained[2] * 100, 1), "%)")
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(plot_kmeans)


plot_true_labels <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster, shape = True_Label)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(
    title = " True values",
    x = paste0("PC1 (", round(variance_explained[1] * 100, 1), "%)"),
    y = paste0("PC2 (", round(variance_explained[2] * 100, 1), "%)")
  ) +
  scale_shape_manual(values = c("0" = 16, "1" = 17)) + 
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(plot_true_labels)
