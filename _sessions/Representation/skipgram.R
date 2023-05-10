download.file("https://snap.stanford.edu/data/finefoods.txt.gz", "finefoods.txt.gz")


library(tidyverse)
library(reticulate)
library(purrr)
library(keras)

reviews <- read_lines("finefoods.txt.gz") 
reviews <- reviews[str_sub(reviews, 1, 12) == "review/text:"]
reviews <- str_sub(reviews, start = 14)
reviews <- iconv(reviews, to = "UTF-8")

reviews = reviews[1:10000]
reviews = reviews[!is.na(reviews)]

head(reviews, 2)

tokenizer <- text_tokenizer(num_words = 500)
tokenizer %>% fit_text_tokenizer(reviews)



embedding_size <- 2  # Dimension of the embedding vector.
skip_window <- 5       # How many words to consider left and right.
num_sampled <- 1       # Number of negative examples to sample for each word.

input_target <- layer_input(shape = 1)
input_context <- layer_input(shape = 1)

embedding <- layer_embedding(
  input_dim = tokenizer$num_words + 1, 
  output_dim = embedding_size, 
  input_length = 1, 
  name = "embedding"
  )


target_vector <- input_target %>% 
  embedding() %>% 
  layer_flatten()

context_vector <- input_context %>%
  embedding() %>%
  layer_flatten()

dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)
output <- layer_dense(dot_product, units = 1, activation = "sigmoid")

model <- keras_model(list(input_target, input_context), output)
model %>% compile(loss = "binary_crossentropy", optimizer = "adam")

summary(model)


skipgrams_generator <- function(text, tokenizer, window_size, negative_samples) {
  gen <- texts_to_sequences_generator(tokenizer, sample(text))
  function() {
    skip <- generator_next(gen) %>%
      skipgrams(
        vocabulary_size = tokenizer$num_words, 
        window_size = window_size, 
        negative_samples = 1
        )
    x <- transpose(skip$couples) %>% map(. %>% unlist %>% as.matrix(ncol = 1))
    y <- skip$labels %>% as.matrix(ncol = 1)
    list(x, y)
    }
  }

# seqs = texts_to_sequences(tokenizer, reviews)
# seqs = seqs[lengths(seqs) > 1]
# #skips = lapply(seqs, skipgrams)
# for(i in 1:length(seqs)) skips[[i]] = skipgrams(seqs[[i]], vocabulary_size = tokenizer$num_words, window_size = 5, negative_samples = 1)
# 
# seqs[[251]]
# 
# skipgrams(seqs[[1]], vocabulary_size = tokenizer$num_words, window_size = 5, negative_samples = 1)
# 
# 
# str(skipgrams_generator(reviews, tokenizer, skip_window, negative_samples)())

model %>%
  fit_generator(
    skipgrams_generator(reviews, tokenizer, skip_window, negative_samples), 
    steps_per_epoch = 1000, epochs = 5
  )

library(dplyr)

embedding_matrix <- get_weights(model)[[1]]


words <- data_frame(
  word = names(tokenizer$word_index), 
  id = as.integer(unlist(tokenizer$word_index))
)

words <- words %>%
  filter(id <= tokenizer$num_words) %>%
  arrange(id)

row.names(embedding_matrix) <- c("UNK", words$word)

a  = text2vec::sim2(embedding_matrix)[-1,-1]

tsne_embed = tsne::tsne(as.dist(-1*a))

plot.new();plot.window(xlim=range(tsne_embed[,1]),ylim=range(tsne_embed[,2]))
text(tsne_embed[,1],tsne_embed[,2],labels=rownames(a)[-1],cex=.5)

plot(a)
which(a[,1]>.7)


sort(a['espresso',])


library(text2vec)

find_similar_words <- function(word, embedding_matrix, n = 5) {
  similarities <- embedding_matrix[word, , drop = FALSE] %>%
    sim2(embedding_matrix, y = ., method = "cosine")
  
  similarities[,1] %>% sort(decreasing = TRUE) %>% head(n)
}



