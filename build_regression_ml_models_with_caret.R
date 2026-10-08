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
#str(model_list)
#model_list$bridge
#model_list$bridge$pred

#compute predictions
petal_preds = predict(model_list, newdata = test_data)
#petal_preds
petal_RMSE = apply(petal_preds, 2, function(x) sqrt(mean((x - test_data$Petal.Width)^2)))
petal_RMSE

#get model correlations
cor(petal_preds)

#identify top-performing models that are uncorrelated 
#assume they are the following two (bridge, glmnet)
#stacking of models is supposed to improve the predictive power
#can also do ensembling instead of stacking - caretEnsemble
#https://zachmayer.github.io/caretEnsemble/
models_for_stack = c("bridge","glmnet")
stack_list = caretList(Petal.Width ~ Sepal.Length + Sepal.Width + Petal.Length,
                       data = train_data,
                       trControl = my_control,
                       methodList = models_for_stack)
model_stack = caretStack(stack_list,
                         method = "glm",
                         trControl = my_control)
summary(model_stack) #combines the two models into one lm - coefficients are in the output

stack_preds = predict(model_stack,newdata = test_data)
stack_preds
RMSE_stack = sqrt(mean((stack_preds$pred - test_data$Petal.Width)^2))
RMSE_stack
petal_RMSE

#best model is cubist; create diagnostic plots
cubist_preds = data.frame(petal_preds[, "cubist"])

plot_data <- data.frame(
  Actual = test_data$Petal.Width,
  Predicted = cubist_preds,
  Residual = round(test_data$Petal.Width - cubist_preds,2)
)
names(plot_data) = c("Actual", "Predicted", "Residual")

# Actual versus predicted
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(size = 3, color = "steelblue") +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "red",
    linetype = "dashed"
  ) +
  theme_minimal() +
  labs(
    title = "Cubist: Actual vs Predicted",
    x = "Actual Petal Width",
    y = "Predicted Petal Width"
  )

# Residuals versus predicted values
ggplot(plot_data, aes(x = Predicted, y = Residual)) +
  geom_point(size = 3, color = "steelblue") +
  geom_hline(
    yintercept = 0,
    color = "red",
    linetype = "dashed"
  ) +
  theme_minimal() +
  labs(
    title = "Cubist Residual Plot",
    x = "Predicted Petal Width",
    y = "Residual"
  )

#variable importance
VI = varImp(model_list$glmnet, scale = FALSE)
plot(VI)


#for classification problems check 
#https://cran.r-project.org/web/packages/caret/vignettes/caret.html
