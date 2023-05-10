require(tidyverse)
require(keras)

digit = readRDS('Representation/1_Data/digit.RDS')

# assign 
c(img_train, digit_train) %<-% digit$train

# reshape & rescale images
img_train <- array_reshape(img_train, c(nrow(img_train), 784))
img_train <- img_train / 255

# expand criterion
digit_train <- to_categorical(digit_train, 10)

# expanded criterion
digit_train[1:3,1:8]

# initialize model
model <- keras_model_sequential() 

# build up model layer by layer
model %>% 
  layer_dense(input_shape = c(784),
              units = 256, 
              activation = 'relu') %>% 
  layer_dense(units = 144, 
              activation = 'relu') %>%
  layer_dense(units = 10, 
              activation = 'softmax')


# set up loss function and optimizer
model %>% compile(
  
  # loss for multi-class prediction
  loss = 'categorical_crossentropy',
  
  # optimizer for weight updating 
  optimizer = "adam",
  
  # metrics to track along
  metrics = c('accuracy')
  )

# Fit neural network to data
history <- model %>% fit(
  
  # training features  and criterion
  img_train, digit_train, 
  
  # number of iterations
  epochs = 10,
  
  # number of samples per update
  batch_size = 32
  )

# FUNS --------

to_img = function(img_long){
  if(length(img_long) %% 1 != 0) stop('image must convertible to square')
  matrix(img_long, 
         nrow=sqrt(length(img_long)), 
         ncol=sqrt(length(img_long)), 
         byrow = T)
  }

norm = function(x) {x = x + abs(min(x)) ; x / max(x)}

show_image = function(img_long){
  img = norm(to_img(img_long))
  img <- t(apply(img, 2, rev))
  par(mar=c(0, 0, 0, 0))
  image(1:nrow(img), 1:ncol(img), img, 
        col = gray((0:255)/255), 
        xaxt = 'n', yaxt = 'n')
  }

show_cor_image = function(image){
  img = norm(image)
  img <- (apply(img, 2, rev))
  par(mar=c(0, 0, 0, 0))
  image(1:nrow(img), 1:ncol(img), img, 
        col = gray((0:255)/255), 
        xaxt = 'n', yaxt = 'n')
  }

get_activation_raw = function(model, input, layer, activation = NULL, bias = TRUE){
  
  relu = function(z) {z[z < 0] = 0; z}
  sigmoid = function(z) {1 / (1 + exp(-1*z))}
  
  cnt = 1
  w = model %>% keras::get_weights()
  z = c(input %*% w[[cnt]])
  if(bias) {cnt = cnt + 1; z = z + w[[cnt]]}
  if(!is.null(activation)) z = get(activation)(z)
  if(layer > 1){
    for(i in 2:layer){
      cnt = cnt + 1
      z = c(z %*% w[[cnt]])
      if(bias) {cnt = cnt + 1; z = z + w[[cnt]]} 
      if(!is.null(activation)) z = get(activation)(z)
      }
    }
  z
  }

show_activation = function(model, input, layer = 1, activation = NULL, bias = TRUE, sorted = FALSE, plot = TRUE){
  z = get_activation_raw(model, input, layer = layer, activation = activation, bias = bias)
  if(sorted){
    ord = order(get_activation_raw(model, 1:length(input), layer, activation = NULL, bias = TRUE))    
    z = z[ord]
    }
  if(plot)  show_image(z) else return(z)
  }

show_weights = function(model, to_layer, bias = FALSE, sorted = TRUE){
  weights = model %>% keras::get_weights()
  if(bias) w = weights[[to_layer * 2 - 1]] else w = weights[[to_layer]]
  
  if(sorted) {
    ord = model %>% get_activation_raw(1:nrow(weights[[1]]), to_layer, "relu")
    w = w[,order(ord)]
  }
  
  par(mar=c(0, 0, 0, 0))
  image(1:nrow(w), 1:ncol(w), w, 
        col = gray((0:255)/255), 
        xaxt = 'n', yaxt = 'n')
  
  }


# IMAGES --------

png('Representation/image/5.png',width=3,height=3,res=300,unit='in')
show_image(img_train[1,])
dev.off()


png('Representation/image/5_1.png',width=3,height=3,res=300,unit='in')
show_activation(model, img_train[1,], 1)
dev.off()

png('Representation/image/5_1_relu.png',width=3,height=3,res=300,unit='in')
show_activation(model, img_train[1,], 1, 'relu')
dev.off()

png('Representation/image/5_2.png',width=3,height=3,res=300,unit='in')
show_activation(model, img_train[1,], 2)
dev.off()

png('Representation/image/5_2_relu.png',width=3,height=3,res=300,unit='in')
show_activation(model, img_train[1,], 2, 'relu')
dev.off()

png('Representation/image/weight_mat.png',width=6,height=6,res=300,unit='in')

weights = model %>% keras::get_weights()
image = weights[[1]]
img = norm(image)
img <- (apply(img, 2, rev))
par(mar=c(.35,2,2,.35))
image(1:ncol(img), 1:nrow(img), t(img), 
      col = gray((0:255)/255), 
      xaxt = 'n', yaxt = 'n')
mtext(c(expression(italic(1)),expression(italic("Pixel")),expression(italic(784))), at = c(4,392,776), side=2, line=.7)
mtext(c(expression(italic(1)),expression(paste(italic(w)^1," = ",italic(z)^1)),expression(italic(256))),side=3,at=c(1,128,252),line=.5)

dev.off()


png('Representation/image/cormat.png',width=6,height=6,res=300,unit='in')
weights = model %>% keras::get_weights()
cor_mat = cor(t(weights[[1]]))
cor_mat[cor_mat == 1] = 0
image = cor_mat
img = norm(image)
img <- (apply(img, 2, rev))
par(mar=c(.35, 2, 2, .35))
image(1:nrow(img), 1:ncol(img), img, 
      col = gray((0:255)/255), 
      xaxt = 'n', yaxt = 'n')
mtext(c(expression(italic(1)),expression(italic("Pixel")),expression(italic(784))), at = c(4,392,776), side=2, line=.7)
mtext(c(expression(italic(1)),expression(italic("Pixel")),expression(italic(784))), at = c(4,392,776), side=3, line=.7)
dev.off()


png('Representation/image/hidden_patterns.png',width=6,height=6,res=300,unit='in')

act = img_train[1:1000,] %*% weights[[1]]
#act = t(apply(act, 1, norm))
dig = digit$train$y[1:1000]
ord = order(dig)
act = act[ord,]
dig = dig[ord]

pos = sapply(split(1:length(dig),dig), mean)

par(mar=c(.5,2,2,.5))
image(1:ncol(act), 1:nrow(act), t(act), 
      col = gray((0:255)/255), 
      xaxt = 'n', yaxt = 'n')
mtext(0:9, at = pos, side=2, las=1, line=.7, font=1)
mtext(c(expression(italic('1')),expression(italic(z)^1),expression(italic('256'))),side=3,at=c(1,128,254),line=.5)

dev.off()


png('Representation/image/cormat_digit.png',width=3.5,height=3.5,res=300,unit='in')

act = img_train[1:1000,] %*% weights[[1]]

cos_mat = cor(t(act))
cos_mat[cos_mat == 1] = 0
a = cmdscale(1-norm(cos_mat))

par(mar=c(0,0,1.2,0))
plot.new();plot.window(range(a[,1]),range(a[,2]))
text(a[,1],a[,2],labels = digit$train$y[1:1000], 
     col = viridis::viridis(10)[digit$train$y[1:1000]+1], cex=.8, font=2)
mtext(expression(italic(z)^1),side=3,cex=1.2)

dev.off()




png('Representation/image/cormat_digit2.png',width=3.5,height=3.5,res=300,unit='in')

relu = function(z) {z[z < 0] = 0; z}
act = t(apply(cbind(img_train[1:1000,],1) %*% rbind(weights[[1]], weights[[2]]),1,relu)) %*% weights[[3]]

cos_mat = cor(t(act))
cos_mat[cos_mat == 1] = 0
a = cmdscale(1-norm(cos_mat))

par(mar=c(0,0,1.2,0))
plot.new();plot.window(range(a[,1]),range(a[,2]))
text(a[,1],a[,2],labels = digit$train$y[1:1000], 
     col = viridis::viridis(10)[digit$train$y[1:1000]+1], cex=.8, font=2)
mtext(expression(italic(z)^2),side=3,cex=1.2)

dev.off()



lin = read_lines('Representation/1_Data/analogies.txt')

cats = unlist(str_extract_all(lin, ":[:print:]+"))

parts = split(lin, cumsum(str_detect(lin, ":[:print:]+")))
names(parts) = (cats)

parts = lapply(parts, function(x){x = do.call(rbind, str_split(x, ' ')); x[-1,]})

ws = unique(unlist(parts))
write_lines(str_to_lower(ws), 'Representation/1_Data/analogy_words.txt')

vecs = read_lines('Representation/1_Data/word_vectors.txt')
vecs = str_split(vecs, ' ')
words = sapply(vecs, function(x) x[1])
vecs = do.call(rbind, lapply(vecs, function(x) x[-1]))
rownames(vecs) = str_to_lower(words)
colnames(vecs) = paste0('node_',1:300)
mode(vecs) = 'numeric'
capitals = parts[[1]]
capitals = tolower(capitals)

capital_vecs = vecs[rownames(vecs) %in% c(capitals),]


cosine = function(vecs) vecs %*% t(vecs) / (sqrt(rowSums(vecs ** 2)) %*% t(sqrt(rowSums(vecs ** 2))))

cos = cosine(vecs)

png('Representation/image/capitals.png',width=5,height=5,res=300,unit='in')
d = cmdscale(1-norm(cos))
par(mar=c(0,0,0,0))
plot.new();plot.window(xlim=range(d[,1])*c(1.15,1.1), ylim=range(d[,2]))
pairs = unique(capitals[,1:2])
for(i in 1:nrow(pairs)){
  x = d[pairs[i,1],]
  y = d[pairs[i,2],]
  lines(c(x[1],y[1]),c(x[2],y[2]),lty=1)
  }
points(d, pch=16, col = 'white',cex=2)
text(d[,1], d[,2], labels = rownames(cos), cex = .9, col = cols, font=1)
cols = viridis::cividis(3)[(rownames(cos) %in% capitals[,1])+1]
dev.off()


get_closest = function(vec, vecs, n = 5) {
  v = matrix(vec,nrow=1) %*% t(vecs) / (sum(vec ** 2) * rowSums(vecs ** 2))
  nam = colnames(v)
  v = c(v) ; names(v) = nam
  sort(v, decreasing = T)[1:n]
  }


capital_vecs = capital_vecs[str_to_lower(c(t(unique(parts[[1]][,1:2])))),]

saveRDS(capital_vecs, 'Representation/1_Data/capital.RDS')

capital_vecs




