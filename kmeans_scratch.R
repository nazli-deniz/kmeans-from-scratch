library(shiny)
library(ggplot2)

# Helper Functions

# assign each point to the closest centroid
assignClusters <- function(data, centroids) {
  cluster_assignments <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    point <- data[i, ]
    distances <- apply(centroids, 1, function(c) sqrt(sum((point - c)^2)))
    cluster_assignments[i] <- which.min(distances)
  }
  return(cluster_assignments)
}

# update centroids points in each cluster
updateCentroids <- function(data, clusters, k) {
  centroids <- matrix(0, nrow = k, ncol = ncol(data))
  for (i in 1:k) {
    cluster_points <- data[clusters == i, , drop = FALSE]
    centroids[i, ] <- colMeans(cluster_points)
  }
  return(centroids)
}

# k-means from scratch

kmeans_iterations <- function(data, k, max_iter = 10, tol = 1e-4) {
  set.seed(123)
  n <- nrow(data)
  init_idx <- sample(1:n, k)
  centroids <- data[init_idx, ]
  
  history <- list()
  
  for (i in 1:max_iter) {
    clusters <- assignClusters(data, centroids)
    history[[i]] <- list(iter = i, centroids = centroids, clusters = clusters)
    
    centroids_new <- updateCentroids(data, clusters, k)
    centroid_shift <- sum((centroids - centroids_new)^2)
    centroids <- centroids_new
    
    if (centroid_shift < tol) {
      cat("Converged at iteration", i, "\n")
      break
    }
  }
  return(history)
}

# Prepare Data
data(iris)
# scale data
iris_scaled <- scale(iris[, 1:4])

pca_result <- prcomp(iris_scaled)
pca_data <- data.frame(pca_result$x[, 1:2])

# Shiny App
ui <- fluidPage(
  titlePanel("K-means from Scratch: Iteration Visualization"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("iter", "Iteration:", min = 1, max = 10, value = 1, step = 1),
      numericInput("k", "Number of Clusters (k):", value = 3, min = 1, max = 10),
      actionButton("run", "Run K-means"),
      br(),
      helpText("Use the slider to visualize how clusters and centroids evolve each iteration.")
    ),
    mainPanel(
      plotOutput("clusterPlot", height = "600px")
    )
  )
)

server <- function(input, output, session) {
  # Run the algorithm when button is clicked
  results <- eventReactive(input$run, {
    kmeans_iterations(iris_scaled, k = input$k, max_iter = 10)
  })
  
  observe({
    hist_len <- length(results())
    updateSliderInput(session, "iter", max = hist_len, value = 1)
  })
  
  output$clusterPlot <- renderPlot({
    hist <- results()
    if (length(hist) == 0) return(NULL)
    
    iter <- input$iter
    data_iter <- hist[[iter]]
    clusters <- as.factor(data_iter$clusters)
    centroids <- data_iter$centroids
    
    # Project centroids into PCA space
    colnames(centroids) <- colnames(iris_scaled)
    centroids_pca <- predict(pca_result, newdata = centroids)
    centroids_pca <- as.data.frame(centroids_pca[, 1:2])
    
    
    pca_data$Cluster <- clusters
    
    ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
      geom_point(size = 3, alpha = 0.7) +
      geom_point(data = centroids_pca, aes(x = PC1, y = PC2),
                 shape = 8, color = "black", size = 6) +
      ggtitle(paste("Iteration", iter, "of K-means (k =", input$k, ")")) +
      theme_minimal(base_size = 15)
  })
}

shinyApp(ui, server)
