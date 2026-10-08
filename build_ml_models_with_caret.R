library(caret)
library(caretEnsemble)
library(dplyr)

set.seed(20261008)
train_index = createDataPartition(
  y = iris$Petal.Width,
  p = 0.8, #in train set
  list = FALSE,
  times = 1
)

train_data = iris[train_index, !names(iris) %in% "Species"]
test_data = iris[-train_index, !names(iris) %in% "Species"]

dim(train_data)

#define 10-fold cross-validation for training/ automatic tuning of the parameters
my_control = trainControl(method = "cv", 
                          number = 10,
                          savePredictions = "final")

#list methods to try for predictions
#list at https://topepo.github.io/caret/available-models.html
method_list = c("bridge","cubist",
                "enet","glmnet","leapSeq",
                "lm","RRF")


#fit training models
model_list = caretList(Petal.Width ~ Sepal.Length + Sepal.Width + Petal.Length,
                       data = train_data,
                       trControl = my_control,
                       methodList = method_list)

#compare the successfully fitted models
resamples <- resamples(model_list)
summary(resamples)
dotplot(resamples, metric = "RMSE")

#evaluate predictions on the test set
results <- lapply(model_list, function(model) {
  pred <- predict(model, newdata = test_data)
  
  postResample(
    pred = pred,
    obs = test_data$Petal.Width
  )
})

do.call(rbind, results)

#best model is cubist, with the smallest RMSE
#however, the model choice is based on the test set only (test RMSE)
#visualize - create a diagnostic plot for the cubist model
library(ggplot2)

pred <- predict(model_list$cubist, newdata = test_data)

comparison <- data.frame(
  Actual = test_data$Petal.Width,
  Predicted = pred
)

ggplot(comparison, aes(x = Actual, y = Predicted)) +
  geom_point(color = "steelblue", size = 3) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "red"
  ) +
  labs(
    title = "Cubist: Actual vs Predicted Petal Width",
    x = "Actual Petal Width",
    y = "Predicted Petal Width"
  ) +
  theme_minimal()

#for a more reliable comparison, examine the 10 fold cross validation results already calculated
#CV RMSE shows the best model to be RRF
resamples <- resamples(model_list)
summary(resamples)
bwplot(resamples, metric = "RMSE")

# Extract cross validation RMSE for each model
cv_results <- sapply(model_list, function(model) {
  min(model$results$RMSE, na.rm = TRUE)
})

# Sort from lowest to highest RMSE
sort(cv_results)

# Identify best model
best_model_name <- names(which.min(cv_results))
best_model_name

#evaluate the selected model on test data
best_model <- model_list[[best_model_name]]

pred <- predict(
  best_model,
  newdata = test_data
)

#evaluate how accurately the trained model (RRF) predicts observed values
postResample(
  pred = pred,
  obs = test_data$Petal.Width
)


#test xgbLinear separately
library(xgboost)
library(caret)

# Define predictors
predictors <- c(
  "Sepal.Length",
  "Sepal.Width",
  "Petal.Length"
)

# Convert predictors to matrices
x_train <- as.matrix(train_data[, predictors])
x_test  <- as.matrix(test_data[, predictors])

# Extract response
y_train <- train_data$Petal.Width
y_test  <- test_data$Petal.Width

# Create XGBoost training data
dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train
)

# Define parameters
params <- list(
  booster = "gblinear",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.3,
  lambda = 0.1,
  alpha = 0.1,
  nthread = 1
)

# Train model
set.seed(20261008)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

# Predict on test data
pred <- predict(
  xgb_model,
  newdata = x_test
)

# Calculate performance
postResample(
  pred = pred,
  obs = y_test
)

#add 10-fold cross-validation to tune and compare models
set.seed(20261008)

cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 200,
  nfold = 10,
  early_stopping_rounds = 20,
  verbose = FALSE
)

# View cross validation results
print(cv_model)

# Best number of boosting rounds
cv_model$early_stop$best_iteration

#use the number of rounds that no longer improve performance to fit the model
best_rounds <- cv_model$early_stop$best_iteration

set.seed(20261008)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_rounds,
  verbose = 0
)

pred <- predict(xgb_model, x_test)

postResample(pred, y_test)

#try xgbTree 
params_tree <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 1
)

set.seed(20261008)

xgb_tree <- xgb.train(
  params = params_tree,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

pred_tree <- predict(xgb_tree, x_test)

postResample(pred_tree, y_test)


#compare XCBoost to the original models
#this comparison code uses the test set for evaluation
#existing caret model results
results <- lapply(model_list, function(model) {
  pred <- predict(model, newdata = test_data)
  
  postResample(
    pred = pred,
    obs = test_data$Petal.Width
  )
})

#add XGBoost linear
results$xgbLinear <- postResample(
  predict(xgb_model, x_test),
  y_test
)

#add XGBoost tree
results$xgbTree <- postResample(
  pred_tree,
  y_test
)

# Compare all models
comparison <- do.call(rbind, results)

comparison[order(comparison[, "RMSE"]), ]


