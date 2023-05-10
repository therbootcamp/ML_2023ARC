# install.packages("keras")
# install.packages("tensorflow")
# tensorflow::install_tensorflow()
library(tidyverse)
library(keras)
source('helper.R')

# Get data -------

# download datasets
fashion_mnist <- dataset_fashion_mnist()

saveRDS(fashion_mnist, 'DeepFeedForward/1_Data/fashion.RDS')

# assign 
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# set labels
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

# Plot digit -------

png('DeepFeedForward/image/fashion1.png',width=8,height=8,res = 300,unit='in')
i  = 1:64
plt_imgs(train_images[i,,], class_names[train_labels[i]+1])
dev.off()

png('DeepFeedForward/image/fashion2.png',width=8,height=8,res = 300,unit='in')
i  = 65:128
plt_imgs(train_images[i,,], class_names[train_labels[i]+1])
dev.off()


# Preprocess data -------

# rescale
train_images <- train_images / 255
test_images <- test_images / 255


# Setup network -------

# layers
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# loss, optimizers, & metrics
model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
  )

# Fit network -------

history <- model %>% fit(
  train_images, train_labels, 
  epochs = 10, validation_split = 0.2
  )

plot(history)


# Evaluate pred -------

# overall performance
model %>% evaluate(test_images, test_labels, verbose = 0)

# predictions 
pred = model %>% predict_classes(test_images)
table(class_names[test_labels+1], class_names[pred+1])

# incorrect 
i = 1:64
plt_imgs(test_images[i,,], class_names[pred[i]+1], class_names[test_labels[i]+1])





