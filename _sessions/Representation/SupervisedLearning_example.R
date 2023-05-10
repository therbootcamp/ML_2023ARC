# Fitting, tune evaluating regression, decision trees, and random forests

# Step 0: Load packages-----------
library(tidyverse)  
library(caret)    
library(partykit) 
library(party)       

# Step 1: Load, Clean, Split, and Explore data ----------------------

# mpg data
data(mpg)

# Explore data
mpg        
View(mpg) 
dim(mpg)   
names(mpg) 

# Convert all characters to factor
mpg <- mpg %>% mutate_if(is.character, factor)

# split index
train_index <- createDataPartition(mpg$hwy, p = .2, list = FALSE)

# train and test sets
data_train <- mpg %>% slice(train_index)
data_test  <- mpg %>% slice(-train_index)

# Step 2: Define training control parameters -------------

# Set method = "none" for now 
ctrl_none <- trainControl(method = "none") 

# Step 3: Train model: -----------------------------

# Regression -------

# Fit model
hwy_glm <- train(form = hwy ~ year + cyl + displ,
                 data = data_train,
                 method = "glm",
                 trControl = ctrl_none)

# Look at summary information
hwy_glm$finalModel
summary(hwy_glm)

# Save fitted values
glm_fit <- predict(hwy_glm)

#  Calculate fitting accuracies
postResample(pred = glm_fit, 
             obs = data_train$hwy)

# Decision Trees -------

# Fit model
hwy_rpart <- train(form = hwy ~ year + cyl + displ,
                   data = data_train,
                   method = "rpart",
                   trControl = ctrl_none,
                   tuneGrid = expand.grid(cp = .01))   # Set complexity parameter

# Look at summary information
hwy_rpart$finalModel
plot(as.party(hwy_rpart$finalModel))   # Visualise your trees

# Save fitted values
rpart_fit <- predict(hwy_rpart)

# Calculate fitting accuracies
postResample(pred = rpart_fit, 
             obs = data_train$hwy)

# Random Forests -------

# fit model
hwy_rf <- train(form = hwy ~ year + cyl + displ,
                data = data_train,
                method = "rf",
                trControl = ctrl_none)

# Look at summary information
hwy_rf$finalModel

# Save fitted values
rf_fit <- predict(hwy_rf)

# Calculate fitting accuracies
postResample(pred = rf_fit, 
             obs = data_train$hwy)


# Step 5: Evaluate prediction ------------------------------

# Define criterion_train
criterion_test <- data_test$hwy

# Save predicted values
glm_pred <- predict(hwy_glm, newdata = data_test)
rpart_pred <- predict(hwy_rpart, newdata = data_test)
rf_pred <- predict(hwy_rf, newdata = data_test)

#  Calculate fitting accuracies
postResample(pred = glm_pred, obs = criterion_test)
postResample(pred = rpart_pred, obs = criterion_test)
postResample(pred = rf_pred, obs = criterion_test)



# Step 6: Modeling tuning ------------------------------

# Use 10-fold cross validation
ctrl_cv <- trainControl(method = "cv", 
                        number = 10) 

# Lasso
lambda_vec <- 10 ^ seq(-3, 3, length = 100)
hwy_lasso <- train(form = hwy ~ year + cyl + displ,
                   data = data_train,
                   method = "glmnet",
                   trControl = ctrl_cv,
                   preProcess = c("center", "scale"),  
                   tuneGrid = expand.grid(alpha = 1,  
                                          lambda = lambda_vec))

# decision tree
cp_vec <- seq(0, .1, length = 100)
hwy_rpart <- train(form = hwy ~ year + cyl + displ,
                   data = data_train,
                   method = "rpart",
                   trControl = ctrl_cv,
                   tuneGrid = expand.grid(cp = cp_vec))

# random forest
mtry_vec <- seq(2, 5, 1)
hwy_rpart <- train(form = hwy ~ year + cyl + displ,
                   data = data_train,
                   method = "rf",
                   trControl = ctrl_cv,
                   tuneGrid = expand.grid(mtry = mtry_vec))

# Step 7: Evaluate tuned models ------------------------------

#  Calculate fitting accuracies
postResample(pred = predict(hwy_lasso, newdata = data_test), 
             obs = criterion_test)
postResample(pred = predict(hwy_rpart, newdata = data_test), 
             obs = criterion_test)
postResample(pred = predict(hwy_rpart, newdata = data_test), 
             obs = criterion_test)


