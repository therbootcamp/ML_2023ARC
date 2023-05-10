require(tidyverse)
cho = read_csv('FeedForwardWithKeras/1_Data/Experiment 1.csv')

probs = readxl::read_excel('FeedForwardWithKeras/1_Data/LotteryProblems.xlsx',sheet='All_Gambles_ToCopy')

As = probs$`Gamble A`[1:91]
Bs = probs$`Gamble B`[1:91]

As = do.call(rbind, str_split(As, '[,;]'))
Bs = do.call(rbind, str_split(Bs, '[,;]'))

colnames(As) = c('o_oA1','o_pA1','o_oA2','o_pA2')
colnames(Bs) = c('o_oB1','o_pB1','o_oB2','o_pB2')

probs = tibble(problem = 1:91) %>% 
  bind_cols(as_tibble(As)) %>% 
  bind_cols(as_tibble(Bs))

cho = cho %>% select(subject, trial, task, boxname, option, cell, celltype, choice)

behav = cho %>% 
  mutate(id = as.numeric(factor(subject))) %>% 
  group_by(id, trial, task) %>% 
  summarize(
    choice = ifelse(first(choice) == 'SpielA', 0, 1),
    n_a1 = sum(boxname == 'a1'),
    n_a2 = sum(boxname == 'a2'),
    n_a3 = sum(boxname == 'a3'),
    n_a4 = sum(boxname == 'a4'),
    n_b1 = sum(boxname == 'b1'),
    n_b2 = sum(boxname == 'b2'),
    n_b3 = sum(boxname == 'b3'),
    n_b4 = sum(boxname == 'b4'),
    n_a = n_a1 + n_a2 + n_a3 + n_a4,
    n_b = n_b1 + n_b2 + n_b3 + n_b4) %>% 
  rename(time = trial,
         problem = task) %>% 
  ungroup() 


one_hot = function(x, n) {v = rep(0, n); v[x] = 1; v}

subj = t(sapply(behav$id, one_hot, n = 90))
colnames(subj) = paste0('s_',1:90)
problems = t(sapply(behav$problem, one_hot, n = 91))
colnames(problems) = paste0('p_',1:91)


data = behav %>% 
  select(id, time, problem, choice) %>% 
  left_join(probs) %>% 
  bind_cols(behav %>% select(-id, time, problem, choice)) %>% 
  bind_cols(as_tibble(subj)) %>% 
  bind_cols(as_tibble(problems)) %>% 
  select(-id, -problem) %>% 
  readr::type_convert()

data_1 = data %>% filter(time == 1) %>% select(-time)
data_2 = data %>% filter(time == 2) %>% select(-time)


pca=princomp(training_feature)


# SPLITTING ---- 

# training sets
training_feature = data_1 %>% select(startss_with('o_'), starts_with('s_')) %>% as.matrix()
training_choice = data_1 %>% select(choice) %>% unlist()

# preiction set
test_feature = data_2 %>% select(starts_with('o_'), starts_with('s_')) %>% as.matrix()
test_choice = data_2 %>% select(choice) %>% unlist()

# layers
net <- keras_model_sequential()
net %>%
  layer_dense(input_shape = ncol(training_feature), units = 20, activation = 'relu') %>% 
  layer_dense(units = 20, activation = 'relu') %>% 
  layer_dense(units = 20, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid') 

# loss, optimizers, & metrics
net %>% compile(
  optimizer = 'adam', 
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
  )

net %>% fit(
  training_feature, 
  training_choice, 
  epochs = 100
  )

# overall performance
net %>% evaluate(test_feature, test_choice, verbose = 0)


get_weights(net)




cor(cbind(training_feature,training_choice))
cor(cbind(test_feature,test_choice))



tr = data_1 %>% select(starts_with('n_'), choice) %>% mutate(choice = factor(choice))
te = data_2 %>% select(starts_with('n_'), choice) %>% mutate(choice = factor(choice))


m = glm(choice ~ ., data = tr %>% select(-n_a4,-n_b4), family = 'binomial')
mean(tr$choice == as.numeric(predict(m, newdata = tr, type = 'response')>.5))
mean(te$choice == as.numeric(predict(m, newdata = te, type = 'response')>.5))


m = caret::train(choice ~ ., 
             data = tr, 
             preProcess = c('scale'), 
             method = 'glmnet',
             tuneGrid = expand.grid(alpha = 0, lambda = .001),
             trControl = caret::trainControl(method='none'))
caret::confusionMatrix(d$choice, predict(m))$overall
caret::confusionMatrix(test$choice, predict(m, newdata = test))$overall


m = caret::train(choice ~ ., 
                 data = tr, 
                 preProcess = c('scale'), 
                 method = 'rf',
                 tuneGrid = expand.grid(mtry=3),
                 trControl = caret::trainControl(method='none'))
caret::confusionMatrix(tr$choice, predict(m))$overall
caret::confusionMatrix(te$choice, predict(m, newdata = te))$overall











