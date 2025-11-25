# 1. FIX: Set Working Directory (Crucial for finding the CSV)
setwd("C:/Users/sg802/Documents/git_clone/hybrid_sim_project")

# 2. FIX: Install and Load tidyr (Needed for replace_na)
if (!requireNamespace("tidyr", quietly = TRUE)) {
    install.packages("tidyr")
}
library(tidyr)

# 3. Install and Load igraph (Needed for plotting)
if (!requireNamespace("igraph", quietly = TRUE)) {
    install.packages("igraph")
}
library(igraph)

# 4. Load dplyr (Needed for mutate)
library(dplyr)

# --- Now run the visualization code ---

# --- Configuration ---
PEDIGREE_FILE <- "C:\\Users\\sg802\\Documents\\git_clone\\hybrid_sim_project\\scripts\\simulation_outputs\\results\\results_rep_3_pedigree.csv"
OUTPUT_PLOT_FILE <- "simulation_outputs_R/results/results_rep_3_pedigree_R_plot.png"
PLOT_TITLE <- "Full Simulation Pedigree"

# Ensure the output directory exists!
output_dir <- dirname(OUTPUT_PLOT_FILE)
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# 1. Load Data
pedigree_df <- read.csv(PEDIGREE_FILE)

# 2. Handle NAs and Clean Edges (using your exact column names: 'parent1_id', 'parent2_id')
pedigree_df <- pedigree_df %>%
  # Use 'parent1_id' and 'parent2_id' for the mutate operation
  mutate(
    parent1_id = replace_na(parent1_id, ""),
    parent2_id = replace_na(parent2_id, "")
  )

# 3. Build Edge List (Relationships)
# Use 'offspring_id' for the 'to' column and the parent IDs for the 'from' column
edges_p1 <- data.frame(from = pedigree_df$parent1_id, to = pedigree_df$offspring_id)
edges_p2 <- data.frame(from = pedigree_df$parent2_id, to = pedigree_df$offspring_id)

edges_df <- rbind(edges_p1, edges_p2) %>%
  filter(from != "" & to != "") %>%
  distinct()

# --------------------------------------------------------
# 4. FIX for "Invalid vertex names": Explicitly Define All Individuals 
# --------------------------------------------------------

# Get a unique list of all individuals (all offspring IDs + all parent IDs)
individuals <- unique(c(pedigree_df$offspring_id, edges_df$from))

# 5. Create the igraph object, defining vertices explicitly
g <- graph_from_data_frame(
    d = edges_df, 
    directed = TRUE,
    vertices = data.frame(name = individuals)
)

# --------------------------------------------------------

# 6. Define Layout and Styling
# Identify the founders (nodes with no incoming edges) to use as tree roots
founders <- V(g)$name[degree(g, mode="in") == 0]

# Use the tree layout, rooted at the founders
l <- layout_as_tree(g, root=founders, mode = "out") 

V(g)$label.color <- "black"
V(g)$shape <- "circle"

# 7. Plot and Save
png(OUTPUT_PLOT_FILE, width = 1200, height = 800, res = 100)
plot(g,
     layout = l,
     vertex.size = 15,
     vertex.color = "lightblue",
     vertex.label.cex = 0.7,
     edge.arrow.size = 0.5,
     main = PLOT_TITLE)
dev.off() 

print(paste0("R Pedigree plot saved to: ", OUTPUT_PLOT_FILE))