require(tidyverse)
require(keras)

# load data
d = readRDS('~/Dropbox (2.0)/Work/Projects/DfE/-- MetaAna/1 Data/_master/des.RDS') %>% as_tibble()

# remove problems with n_outcomes > 2
prob = d %>% select_if(str_detect(names(d), '[AB][345]'))
sel = apply(prob, 1, function(x) any(x != 0))
d = d %>% 
  filter(!sel) %>% 
  select_if(!str_detect(names(d), '[AB][345]'))

# Determine risky choice
variance = function(x) {
  ev = x[1] * x[2] + x[3] * x[4]
  x[1]**2 * x[2] + x[3]**2 * x[4] - ev**2}
As = d %>% select_if(str_detect(names(d), '[A][12]'))
Bs = d %>% select_if(str_detect(names(d), '[B][12]'))
d$var0 = apply(As, 1, variance)
d$var1 = apply(Bs, 1, variance)
risky = as.numeric(d$var0 < d$var1)

# Recode problems
probs = matrix(nrow=nrow(d),ncol=8)
colnames(probs) = c('outA1','probA1','outA2','probA2',
                    'outB1','probB1','outB2','probB2')
for(i in 1:nrow(d)){
  if(is.na(risky[i]) || risky[i] == 0){
    if(d$outA1[i]>d$outA2[i]){
      probs[i, 1] = d$outA1[i]
      probs[i, 2] = d$probA1[i]
      probs[i, 3] = d$outA2[i]
      probs[i, 4] = d$probA2[i]
      } else {
      probs[i, 1] = d$outA2[i]
      probs[i, 2] = d$probA2[i]
      probs[i, 3] = d$outA1[i]
      probs[i, 4] = d$probA1[i]
      }
    if(d$outB1[i]>d$outB2[i]){
      probs[i, 5] = d$outB1[i]
      probs[i, 6] = d$probB1[i]
      probs[i, 7] = d$outB2[i]
      probs[i, 8] = d$probB2[i]
      } else {
      probs[i, 5] = d$outB2[i]
      probs[i, 6] = d$probB2[i]
      probs[i, 7] = d$outB1[i]
      probs[i, 8] = d$probB1[i]
      }  
    } else {
      if(d$outA1[i]>d$outA2[i]){
      probs[i, 5] = d$outA1[i]
      probs[i, 6] = d$probA1[i]
      probs[i, 7] = d$outA2[i]
      probs[i, 8] = d$probA2[i]
      } else {
      probs[i, 5] = d$outA2[i]
      probs[i, 6] = d$probA2[i]
      probs[i, 7] = d$outA1[i]
      probs[i, 8] = d$probA1[i]
      }
      if(d$outB1[i]>d$outB2[i]){
      probs[i, 1] = d$outB1[i]
      probs[i, 2] = d$probB1[i]
      probs[i, 3] = d$outB2[i]
      probs[i, 4] = d$probB2[i]
      } else {
      probs[i, 1] = d$outB2[i]
      probs[i, 2] = d$probB2[i]
      probs[i, 3] = d$outB1[i]
      probs[i, 4] = d$probB1[i]
      }        
    }
  }

# remove redundant probability
probs = probs[, -c(4,8)]

# recode choice and cpt prediction
choice = ifelse(risky == 1, d$choice, abs(1-d$choice))
cpt = ifelse(risky == 1, d$cpt, abs(1-d$cpt))

# mean(choice == cpt)

# create second problem set with cpt
probs_cpt = probs %>% cbind(cpt)


# SPLIT --------

# set training index
set.seed(100)
training = sample(length(choice), length(choice) * .8)

# training sets
training_prob = probs[training, ]
training_cpt_prob = probs_cpt[training, ]
training_choice = (choice[training])

# preiction sets
test_prob = probs[-training,]
test_cpt_prob = probs_cpt[-training,]
test_choice = (choice[-training])


# SETUP NEURAL NETWORKS --------

# layers
model1 <- keras_model_sequential()
model1 %>%
  layer_dense(units = 64, input_shape = 6, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid') 

# layers
model2 <- keras_model_sequential()
model2 %>%
  layer_dense(units = 64, input_shape = 7, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid') 

# layers
model3 <- keras_model_sequential()
model3 %>%
  layer_dense(units = 32, input_shape = 6, activation = 'relu') %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 8, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# loss, optimizers, & metrics
model1 %>% compile(
  optimizer = 'adam', 
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
  )

# loss, optimizers, & metrics
model2 %>% compile(
  optimizer = 'adam', 
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
  )

# loss, optimizers, & metrics
model3 %>% compile(
  optimizer = 'adam', 
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)


# RUN NEURAL NETWORKS --------

model1 %>% fit(
  training_prob, 
  training_choice, 
  epochs = 40
  )

model2 %>% fit(
  training_cpt_prob, training_choice, 
  epochs = 40
  )

model3 %>% fit(
  training_prob, training_choice, 
  epochs = 40
  )

# EVALUATE NEURAL NETWORKS --------

# overall performance
model1 %>% evaluate(test_prob, test_choice, verbose = 0)

# overall performance
model2 %>% evaluate(test_cpt_prob, test_choice, verbose = 0)

# overall performance
model3 %>% evaluate(test_prob, test_choice, verbose = 0)


# SETUP FOR CARET --------

tbl_train = as_tibble(training_prob) %>% bind_cols(tibble(choice = training_choice))
tbl_cpt_train = as_tibble(training_cpt_prob) %>% bind_cols(tibble(choice = training_choice))

tbl_test = as_tibble(test_prob) %>% bind_cols(tibble(choice = test_choice))
tbl_cpt_test = as_tibble(test_cpt_prob) %>% bind_cols(tibble(choice = test_choice))


# RUN RANDOM FORESTS --------

rf1 = caret::train(factor(choice) ~ ., data = tbl_train, 
                   method='rf', 
                   trControl = trainControl(method='none'))
mean(predict(rf1) == tbl_train$choice)
mean(predict(rf1, newdata = tbl_test) == tbl_test$choice)

rf2 = caret::train(factor(choice) ~ ., data = tbl_cpt_train, 
                   method='rf', 
                   trControl = trainControl(method='none'))
mean(predict(rf2) == tbl_cpt_train$choice)
mean(predict(rf2, newdata = tbl_cpt_test) == tbl_cpt_test$choice)


# RUN LOGISTIC REGRESSIONS --------

lm1 = caret::train(factor(choice) ~ ., data = tbl_train, 
                   method='glm', 
                   trControl = trainControl(method='none'))
mean(predict(lm1) == tbl_train$choice)
mean(predict(lm1, newdata = tbl_test) == tbl_test$choice)

lm2 = caret::train(factor(choice) ~ ., data = tbl_cpt_train, 
                   method='glm', 
                   trControl = trainControl(method='none'))
mean(predict(lm2) == tbl_cpt_train$choice)
mean(predict(lm2, newdata = tbl_cpt_test) == tbl_cpt_test$choice)


