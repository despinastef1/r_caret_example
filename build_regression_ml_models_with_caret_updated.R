library(caret)
library(caretEnsemble)
library(xgboost)
library(dplyr)

set.seed(20261008)

#create training and test data sets
train_index <- createDataPartition(
  y = iris$Petal.Width,
  p = 0.8,
  list = FALSE
)

train_data <- iris[
  train_index,
  !names(iris) %in% "Species"
]

test_data <- iris[
  -train_index,
  !names(iris) %in% "Species"
]

#create identical folds for all models
set.seed(20261008)

folds <- createFolds(
  train_data$Petal.Width,
  k = 10,
  returnTrain = FALSE
)

# Training indices for caret
train_folds <- lapply(
  folds,
  function(i) setdiff(seq_len(nrow(train_data)), i)
)

my_control <- trainControl(
  method = "cv",
  number = 10,
  index = train_folds,
  indexOut = folds,
  savePredictions = "final",
  allowParallel = FALSE
)

#train and tune the chosen caret models
#list at https://topepo.github.io/caret/available-models.html

method_list <- c(
  "bridge",
  "cubist",
  "enet",
  "glmnet",
  "leapSeq",
  "lm",
  "RRF"
)

set.seed(20261008)

model_list <- caretList(
  Petal.Width ~ Sepal.Length + Sepal.Width + Petal.Length,
  data = train_data,
  trControl = my_control,
  methodList = method_list,
  metric = "RMSE"
)

# Extract the selected tuning configuration's
# cross validation RMSE for each model
caret_cv <- sapply(model_list, function(model) {
  min(model$results$RMSE, na.rm = TRUE)
})

caret_cv

#tune xgboost linear and xgboost tree

#prepare the data
predictors <- c(
  "Sepal.Length",
  "Sepal.Width",
  "Petal.Length"
)

x_train <- as.matrix(train_data[, predictors])
x_test <- as.matrix(test_data[, predictors])

y_train <- train_data$Petal.Width
y_test <- test_data$Petal.Width

dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train
)

#create a tuning grid and run cross-validation
#candidate XGBoost configurations
xgb_grid <- expand.grid(
  booster = c("gblinear", "gbtree"),
  eta = c(0.05, 0.1, 0.3),
  lambda = c(0, 0.1),
  alpha = c(0, 0.1),
  max_depth = c(2, 3)
)

#max_depth doesn't apply to gblinear
xgb_grid <- xgb_grid[
  xgb_grid$booster == "gbtree" |
    xgb_grid$max_depth == 2,
]

xgb_grid <- unique(xgb_grid)
rownames(xgb_grid) <- NULL

#store tuning results
xgb_results <- data.frame()

set.seed(20261008)

for (i in seq_len(nrow(xgb_grid))) {
  
  config <- xgb_grid[i, ]
  
  params <- list(
    booster = as.character(config$booster),
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = config$eta,
    lambda = config$lambda,
    alpha = config$alpha,
    nthread = 1
  )
  
  if (config$booster == "gbtree") {
    params$max_depth <- config$max_depth
  }
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 200,
    folds = unname(folds),
    nfold = 10,
    early_stopping_rounds = 20,
    verbose = FALSE
  )
  
  best_rounds <- cv$early_stop$best_iteration
  
  rmse <- cv$evaluation_log$test_rmse_mean[best_rounds]
  
  xgb_results <- rbind(
    xgb_results,
    data.frame(
      config,
      nrounds = best_rounds,
      RMSE = rmse
    )
  )
}

#best configurations for each booster
xgb_linear_best <- xgb_results |>
  filter(booster == "gblinear") |>
  arrange(RMSE) |>
  slice(1)

xgb_tree_best <- xgb_results |>
  filter(booster == "gbtree") |>
  arrange(RMSE) |>
  slice(1)

print(xgb_linear_best)
print(xgb_tree_best)

#compare all 9 models using cv
comparison_cv <- data.frame(
  Model = c(
    names(caret_cv),
    "xgbLinear",
    "xgbTree"
  ),
  CV_RMSE = c(
    as.numeric(caret_cv),
    xgb_linear_best$RMSE,
    xgb_tree_best$RMSE
  )
)

comparison_cv <- comparison_cv |>
  arrange(CV_RMSE)

print(comparison_cv)

#select model with lowest CV RMSE
best_model_name <- comparison_cv$Model[1]

cat("Best model:", best_model_name, "\n")

#fit the selected model and evaluate it on the test set
if (best_model_name %in% c("xgbLinear", "xgbTree")) {
  
  best_config <- if (best_model_name == "xgbLinear") {
    xgb_linear_best
  } else {
    xgb_tree_best
  }
  
  best_params <- list(
    booster = as.character(best_config$booster),
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = best_config$eta,
    lambda = best_config$lambda,
    alpha = best_config$alpha,
    nthread = 1
  )
  
  if (best_config$booster == "gbtree") {
    best_params$max_depth <- best_config$max_depth
  }
  
  set.seed(20261008)
  
  final_model <- xgb.train(
    params = best_params,
    data = dtrain,
    nrounds = best_config$nrounds,
    verbose = 0
  )
  
  final_predictions <- predict(
    final_model,
    newdata = x_test
  )
  
} else {
  
  #caret models already refit their best
  #configurations using all training data
  final_model <- model_list[[best_model_name]]
  
  final_predictions <- predict(
    final_model,
    newdata = test_data
  )
}

#final evaluation
final_metrics <- postResample(
  pred = final_predictions,
  obs = y_test
)

print(final_metrics)

#plot actual versus predicted values
library(ggplot2)

evaluation <- data.frame(
  Actual = y_test,
  Predicted = final_predictions
)

ggplot(evaluation, aes(Actual, Predicted)) +
  geom_point(size = 3, color = "steelblue") +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "red"
  ) +
  labs(
    title = paste("Final Model:", best_model_name),
    x = "Actual Petal Width",
    y = "Predicted Petal Width"
  ) +
  theme_minimal()

#variable importance for the selected model (cubist)

#extract best model
best_model <- model_list$cubist

#calculate variable importance
importance <- varImp(best_model, scale = TRUE)

print(importance)

#visualize using ggplot
importance_df <- varImp(model_list$cubist)$importance

importance_df$Variable <- rownames(importance_df)

ggplot(
  importance_df,
  aes(x = reorder(Variable, Overall), y = Overall)
) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Cubist Variable Importance",
    x = "Predictor",
    y = "Importance"
  ) +
  theme_minimal()

#partial dependence plots reveal how predictions change as each variable changes
#install.packages("pdp")
library(pdp)

partial(
  object = best_model,
  pred.var = "Petal.Length",
  train = train_data[, c(
    "Sepal.Length",
    "Sepal.Width",
    "Petal.Length"
  )],
  plot = TRUE,
  rug = TRUE
)

#permutation importance
#the permutation method measures how much prediction error increases when a predictor's values are shuffled
# install.packages(
#   "vip",
#   repos = c(
#     "https://bgreenwell.r-universe.dev",
#     "https://cloud.r-project.org"
#   )
# )
library(vip)

#cubist model specific importance
vip(model_list$cubist, num_features = 3)

#permutation importance
importance <- vi(
  model_list$cubist,
  method = "permute",
  train = train_data,
  target = "Petal.Width",
  metric = "rmse",
  pred_wrapper = function(object, newdata) {
    predict(object, newdata = newdata)
  },
  nsim = 30
)

ggplot(
  importance,
  aes(
    x = Importance,
    y = reorder(Variable, Importance)
  )
) +
  geom_col(fill = "steelblue") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(angle = 45, hjust = 1)
  ) +
  labs(
    title = "Cubist Permutation Importance",
    x = "Increase in RMSE",
    y = "Predictor"
  )

