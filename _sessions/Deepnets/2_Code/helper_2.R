plot_embedding = function(act){
  par(mar=c(0,0,0,0))
  image(1:ncol(act), 1:nrow(act), t(act), 
        col = gray((0:255)/255), 
        xaxt = 'n', yaxt = 'n')
}

plot_cosine = function(cos){
  par(mar=c(0,0,0,0))
  cos = (apply(cos, 2, rev))
  image(1:ncol(cos), 1:nrow(cos), t(cos), 
        col = gray((0:255)/255), 
        xaxt = 'n', yaxt = 'n')
}

cosine = function(vecs) vecs %*% t(vecs) / (sqrt(rowSums(vecs ** 2)) %*% sqrt(t(rowSums(vecs ** 2))))

plot_cosine_mds = function(cos, items, col = TRUE){
  norm = function(x) {x = x + abs(min(x)) ; x / max(x)}
  a = cmdscale(1-norm(cos))
  par(mar=c(0,0,0,0))
  if(col) cols = viridis::viridis(length(unique(items))) else cols=rep('black', length(unique(items)))
  names(cols) = unique(items)
  plot.new();plot.window(range(a[,1]),range(a[,2]))
  text(a[,1],a[,2],labels = items, 
       col = cols[items], cex=.5, font=1)
  #mtext(expression(italic(z)^2),side=3,cex=1.2)
}
