source("kmeans_main.R")

library(shiny)
library(ggplot2)

# Prepare data for Shiny
iris_scaled <- scale(iris[, 1:4])
pca_result <- prcomp(iris_scaled, center = TRUE, scale. = TRUE)

pca_data <- data.frame(
  pca_result$x[, 1:2]
)


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
