
plt = function(img){
  image_1 <- as.data.frame(img)
  colnames(image_1) <- seq_len(ncol(image_1))
  image_1$y <- seq_len(nrow(image_1))
  image_1 <- gather(image_1, "x", "value", -y)
  image_1$x <- as.integer(image_1$x)
  ggplot(image_1, aes(x = x, y = y, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black", na.value = NA) +
    scale_y_reverse() +
    theme_minimal() +
    theme(panel.grid = element_blank())   +
    theme(aspect.ratio = 1) +
    xlab("") +
    ylab("")
}


plt_imgs = function(imgs, labs, true_labs = NULL){
  rows = ceiling(sqrt(nrow(imgs)))
  cols = ceiling(nrow(imgs) / rows)
  
  par(mfcol=c(rows, cols))
  par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
  for (i in 1:nrow(imgs)) { 
    img <- imgs[i, , ]
    img <- t(apply(img, 2, rev)) 
    image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n')
    if(is.null(true_labs)) {
      mtext(paste(labs[i]),cex=.75,line=.2)
      } else {
      col = ifelse(labs[i] == true_labs[i], 'black', 'red')
      lab =  ifelse(labs[i] == true_labs[i], labs[i], paste(labs[i],'(',true_labs[i],')'))
      mtext(lab, col = col,cex=.75,line=.2)
      }
    }
  }

