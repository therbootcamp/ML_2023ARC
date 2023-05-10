library(tidyverse)
library(keras)
source("2_Code/helper.R")


  

# read data set
fashion <- readRDS(file = "1_Data/fashion.RDS")

str(fashion)

# PREPROCESS -------

# split digit  train
c(fashion_train_images, fashion_train_items) %<-% fashion$train

# split digit  test
c(fashion_test_images, fashion_test_items) %<-% fashion$test

# reshape images
fashion_train_images_serialized <- array_reshape(fashion_train_images, 
                                                 c(nrow(fashion_train_images), 784))
fashion_test_images_serialized <- array_reshape(fashion_test_images, 
                                                c(nrow(fashion_test_images), 784))

# rescale images
fashion_train_images_serialized <- fashion_train_images_serialized / 255
fashion_test_images_serialized <- fashion_test_images_serialized / 255

# expand criterion
fashion_train_items_onehot <- to_categorical(fashion_train_items, 10)
fashion_test_items_onehot <- to_categorical(fashion_test_items, 10)

# fashion items
fashion_labels = c('T-shirt/top',
                   'Trouser',
                   'Pullover',
                   'Dress',
                   'Coat', 
                   'Sandal',
                   'Shirt',
                   'Sneaker',
                   'Bag',
                   'Ankle boot')

plt_imgs(fashion_train_images[1:25,,],
         fashion_labels[fashion_train_items[1:25]+1])


# MODEL -------

# begin building network
net <- keras_model_sequential()

# add layer
net %>%  
  layer_dense(
    input_shape = 784,
    units = 256,
    activation = 'relu'
  ) %>% 
  layer_dense(
    units = 144,
    activation = 'relu'
  ) %>%
  layer_dense(
    units = 10,
    activation = "softmax"
  ) 

# loss, optimizers, & metrics
net %>% compile(
  optimizer = 'adam', 
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
  )

# loss, optimizers, & metrics
net %>% fit(
  x = fashion_train_images_serialized[1:100,], 
  y = fashion_train_items_onehot[1:100,],
  batch_size = 32,
  epochs = 100
  )

# evaluate
net %>% evaluate(fashion_test_images_serialized,
                 fashion_test_items_onehot, verbose = 0)

# compare predictions to truth
pred = net %>% predict_classes(fashion_test_images_serialized)
table(fashion_labels[fashion_test_items+1], fashion_labels[pred+1])

