## 1. Load packages
library(tidyverse)
library(cowplot)
library(cols4all)
library(mgcv)
library(GWmodel)
library(broom)
library(spdep)
library(stringr)

set.seed(123) ## example number, it is the same number to the python code

size <- 50

X1 <- runif(n = size*size, min = -1.5, max = 1.5)
X2 <- runif(n = size*size, min = 0, max = 3)
X3 <- runif(n = size*size, min = -1.5, max = 1.5)
X4 <- runif(n = size*size, min = -1.5, max = 1.5)

X <- cbind(X1, X2, X3, X4)

u <- rep(0:(size-1), times = size)
v <- rep(0:(size-1), each = size)

coords_matrix <- cbind(u, v) 

f0 <- 2
f1 <- 2 * X1
f2 <- log(X2 / 2) * X2
f3 <- X3^3 

f4 <- numeric(length(X4))
mask_left <- u < (size / 2)
mask_right <- !mask_left
f4[mask_left] <- X4[mask_left]^3
f4[mask_right] <- -X4[mask_right]^3

err <- runif(n = size*size, min = -1.5, max = 1.5)

y_vector <- f0 + f1 + f2 + f3 + f4 + err

tryCatch({
  loaded_data <- data.frame(
    y = gen_y_vector,
    x1 = gen_X1,
    x2 = gen_X2,
    x3 = gen_X3,
    x4 = gen_X4,
    lng = gen_u,
    lat = gen_v
  )
  
  required_cols <- c("y", "x1", "x2", "x3", "x4", "lng", "lat")
  if (!all(required_cols %in% colnames(loaded_data))) {
    stop("Internal error: Not all required columns were created in the synthetic data.")
  }
  
  cat("Dimensions of loaded_data:", dim(loaded_data)[1], "rows,", dim(loaded_data)[2], "columns.\n")
  cat("First few rows of loaded_data:\n")
  print(head(loaded_data))
  
}, error = function(e) {
  cat("Error during synthetic data structuring: ", e$message, "\n")
  stop("Synthetic data generation or structuring failed.")
})

# Rename columns and add necessary ones for the GAM model
model_data <- loaded_data %>%
  select(
    y = y,         # Response variable
    X1 = x1,       # Predictor 1
    X2 = x2,       # Predictor 2
    X3 = x3,       # Predictor 3
    X4 = x4,       # Predictor 4
    u = lng,       # Spatial coordinate 1 (mapped from lng)
    v = lat        # Spatial coordinate 2 (mapped from lat)
  ) %>%
  mutate(
    Intercept = 1 # Column of 1s for the spatially varying intercept
    # Add true_y and true_signal if they were in the CSV and you want them for comparison
    # true_y = true_y,
    # true_signal = true_signal
  )

# Get the number of locations from the loaded data
n_locations <- nrow(model_data)
cat("Number of data points loaded:", n_locations, "\n")

## 9. Fit the GGP-GAM model with SVCs for all predictors
# Model structure: y = beta0(u,v) + beta1(u,v)*X1 + ... + beta5(u,v)*X5 + error
gam_formula_svc_all <- as.formula(
  y ~ 0 +                      
    s(u, v, bs='gp', by=Intercept) + 
    s(u, v, bs='gp', by=X1) +       
    s(u, v, bs='gp', by=X2) +       
    s(u, v, bs='gp', by=X3) +       
    s(u, v, bs='gp', by=X4)         
)

cat("Fitting GGP-GAM model with SVCs for all predictors...\n")
gam_model_svc_all <- gam(gam_formula_svc_all, data = model_data, method = "REML")
cat("Model fitting complete.\n")

svc_vars <- c("Intercept", "X1", "X2", "X3", "X4")
# Prepare base newdata with all relevant predictor columns set to 0
base_newdata_preds_svc <- model_data %>%
  select(u, v, all_of(svc_vars)) %>% # Select coords + all SVC predictor columns
  mutate(across(all_of(svc_vars), ~ 0)) # Set all these columns to 0
# Add columns for estimated SVCs to the model_data dataframe
for (var_name in svc_vars) {
  cat("Extracting estimated SVC for:", var_name, "...\n")
  # Create specific newdata for this variable
  current_newdata <- base_newdata_preds_svc %>%
    mutate(!!sym(var_name) := 1) # Set the target variable to 1
  
  # Predict the SVC surface
  col_name <- paste0("estimated_beta_", var_name)
  model_data[[col_name]] <- predict(gam_model_svc_all, newdata = current_newdata, type = 'response')
}


# 2) Prediction values
model_data$predicted_y_svc_all <- predict(gam_model_svc_all, newdata = model_data, type = 'response')

## Save Location-based Results to CSV
# This includes coordinates, input variables, true y, true signal, estimated SVCs, and predicted y.
csv_output_filename <- "ggp_gam_svc_all_location_results.csv"
cat("\nSaving location-based results to:", csv_output_filename, "...\n")
write.csv(model_data, csv_output_filename, row.names = FALSE)
cat("Location-based results saved successfully.\n")

## Save Model Summary and Bandwidths to TXT
txt_output_filename <- "ggp_gam_svc_all_model_summary.txt"
cat("\nSaving model summary and bandwidths to:", txt_output_filename, "...\n")

# Use sink() to redirect output to the text file
sink(txt_output_filename)

cat("--- GGP-GAM (SVCs for All Predictors) Model Summary and Parameters ---\n\n")

cat("Model Formula:\n")
print(gam_formula_svc_all)
cat("\n")

cat("Full Model Summary:\n")
print(summary(gam_model_svc_all))
cat("\n")

cat("Smoothing Parameters (SP) for GP smooths (Bandwidths):\n")
# Get SP values and their names (smooth term labels)
sp_values <- gam_model_svc_all$sp
sp_labels <- sapply(gam_model_svc_all$smooth, function(x) x$label) # Labels like s(u,v):Intercept, s(u,v):X1 etc.
sp_df <- data.frame(Label = sp_labels, SP = sp_values)
print(sp_df)
cat("\n")


# Calculate and Print Overall Model Fit Metrics (Predicted y vs True y)
rsq <- function (x, y) cor(x, y) ^ 2
rmse <- function(x, y) sqrt(mean((x - y)^2))

cat("Overall Model Fit Metrics (Predicted y vs True y):\n")
cat("R-squared:", round(rsq(model_data$y, model_data$predicted_y_svc_all), 4), "\n")
cat("RMSE:", round(rmse(model_data$y, model_data$predicted_y_svc_all), 4), "\n")
cat("\n")

# --- Explanation for Non-linear Curves ---
cat("Regarding 'non-linear function relationship curves':\n")
cat("In this specific GGP-GAM model structure (using s(u,v, bs='gp', by=X_i) for all predictors),\n")
cat("the effect of each variable X_i is modeled as X_i * beta_i(u,v).\n")
cat("At any fixed location (u,v), this is a linear relationship with X_i, with the slope being beta_i(u,v).\n")
cat("The non-linearity in the model arises from the spatial variability of the coefficients beta_i(u,v),\n")
cat("which are 2D surfaces over the (u,v) coordinates, not 1D curves plotting the response vs X_i.\n")
cat("Therefore, standard 1D non-linear relationship curves (like partial effect plots for s(X_i) terms)\n")
cat("are not outputs of this model structure.\n")
cat("The estimated 'function relationship' for each variable X_i is its estimated spatial coefficient surface beta_i(u,v).\n")
 cat(paste0("These estimated SVC surfaces are saved as columns ('estimated_beta_", paste(svc_vars, collapse=", estimated_beta_"), "') in the CSV output file.\n"))
cat("\n")


cat("--- End of Model Summary ---\n")

# Stop redirecting output and close the file
sink()

cat("Model summary and bandwidths saved successfully to:", txt_output_filename, "\n")
