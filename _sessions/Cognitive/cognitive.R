require(tidyverse)
fl = readRDS('~/Dropbox (2.0)/Work/Projects/Memory/-- AgingLexicon/2 Clean Data/Tablet_Fluency.RDS')
# cats = read_delim('~/Dropbox (2.0)/Work/Projects/Memory/-- AgingLexicon/0 Material/TroyerCategories_german.txt',col_names = F,delim=',[:blank:]')
# cats$X2 = str_to_lower(str_remove_all(cats$X2 , '[:blank:]'))
# cats = cats %>% na.omit()

fl = fl %>% filter(!is.na(correct), !str_detect(correct, ' '), correct != 'NA') 
fluency = split(fl$correct, fl$subject)
words = fl %>% pull(correct)

# w_sel = unique(words)[(unique(words) %in% cats[[2]])]
# fluency = lapply(fluency, function(x) unique(x[x %in% w_sel]))

write_lines(unique(unlist(fluency)), 'Cognitive/1_Data/anis.txt')

vecs = read_lines('Cognitive/1_Data/animal_vectors.txt')
vecs = str_split(vecs, ' ')
words = str_to_lower(sapply(vecs, function(x) x[1]))
vecs = do.call(rbind, lapply(vecs, function(x) x[-1]))
rownames(vecs) = str_to_lower(words)
colnames(vecs) = paste0('node_',1:300)
mode(vecs) = 'numeric'

fluency = lapply(fluency, function(x) x[x %in% rownames(vecs)])

cosine = function(vecs) vecs %*% t(vecs) / (sqrt(rowSums(vecs ** 2)) %*% t(sqrt(rowSums(vecs ** 2))))

cos_vec = cosine(vecs)
cos_vec[cos_vec == 1] = 0
norm = function(x) {x = x + abs(min(x)) ; x / max(x)}
cos_vec = norm(cos_vec)



plot_cosine_mds = function(cos, items){
    ani_cat = unique(cats$X1)
    col = viridis::viridis(length(ani_cat))
    names(col) = ani_cat
    cols = col[cats$X1]
    names(cols) = cats$X2
    norm = function(x) {x = x + abs(min(x)) ; x / max(x)}
    mds <<- cmdscale(1-norm(cos))
    par(mar=c(0,0,0,0))
    plot.new();plot.window(range(mds[,1]),range(mds[,2]))
    sel = items %in% names(cols)
    text(mds[!sel,1],mds[!sel,2],labels = items[!sel], 
         col = 'grey75', cex=.4, font=1)
    text(mds[sel,1],mds[sel,2],labels = items[sel], 
         col = cols[items[sel]], cex=.5, font=2)
    #mtext(expression(italic(z)^2),side=3,cex=1.2)
  }

add_path = function(x){
  pos = get("mds", pos = '.GlobalEnv')[x,]
  cols = colorRampPalette(c('black','steelblue'))(length(x))
  points(pos, pch = 16, cex=.5, col = cols)
  for(i in 1:(length(x)-1)){
    lines(c(pos[i,1],pos[i+1,1]),c(pos[i,2],pos[i+1,2]), col=cols[i], lwd=1) 
    }
  }

png('Cognitive/image/wordspace.png',width=6,height=6,res=300,unit='in')
plot_cosine_mds(cos_vec, rownames(cos_vec))
dev.off()

png('Cognitive/image/wordspace1.png',width=6,height=6,res=300,unit='in')
plot_cosine_mds(cos_vec, rownames(cos_vec))
add_path(fluency[[1]])
dev.off()

png('Cognitive/image/wordspace2.png',width=6,height=6,res=300,unit='in')
plot_cosine_mds(cos_vec, rownames(cos_vec))
add_path(fluency[[2]])
dev.off()

png('Cognitive/image/wordspace3.png',width=6,height=6,res=300,unit='in')
plot_cosine_mds(cos_vec, rownames(cos_vec))
add_path(fluency[[3]])
dev.off()



require(tidyverse)
require(keras)

digit = readRDS('Representation/1_Data/digit.RDS')

# assign 
c(img_train, digit_train) %<-% digit$train

# reshape & rescale images
img_train <- array_reshape(img_train, c(nrow(img_train), 784))
img_train <- img_train / 255





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


signify = function(x) sign(x - .3)

img1 = signify(img_train[1,])
img2 = signify(img_train[2,])

png('Cognitive/image/5.png',width=3,height=3,res=300,unit='in')
show_image(img1)
dev.off()

png('Cognitive/image/0.png',width=3,height=3,res=300,unit='in')
show_image(img2)
dev.off()


w = img1 %*% t(img1) + img2 %*% t(img2)

img1_noise = signify(img1 + rnorm(length(img1), 0,10))
img2_noise = signify(img1 + rnorm(length(img1), 0,10))

png('Cognitive/image/0.png',width=3,height=3,res=300,unit='in')

show_image(img1_noise)
show_image(signify(img1_noise %*% w))

show_image(recov(img1_noise, 3))

show_image(signify(signify(img %*% w) %*% w))


recov = function(x, n = 1) {
  for(i in 1:n){
    x = signify(x %*% w)
    }
  x
  }

i = 8
show_image(signify(img_train[i,]))
recov = signify(img_train[i,] %*% w)
show_image(-recov)
for(i in 1:3) recov = sign(recov %*% w)
show_image(-recov)





a = w %*% img_train[1,]


show_image(img_train[1,])
show_image(w %*% img_train[1,])

sigmoid = function(x) 1 / (1 + exp(-x))


for(i in 1:4) recov = sign(w %*% recov)
show_image(recov)





