setwd("E:/Competition/xf/UserPersona/")

t0 <- Sys.time()
train <- fread(input = "./data/train.txt", sep = ",", 
               header = FALSE, encoding = "UTF-8") %>% as_tibble()
test  <- fread(input = "./data/apply_new.txt", sep = ",", 
               header = FALSE, encoding = "UTF-8") %>% as_tibble()
t1 <- Sys.time()
cat("read data with data.table::fread() takes ", t1 - t0, "secs.")

colnames(train) <- c("pid", "label", "gender", "age",
                     "tagid", "time", "province", "city",
                     "model", "make")
colnames(test) <- c("pid", "gender", "age",
                    "tagid", "time", "province", "city",
                    "model", "make")

# iconv(train$model, from = "UTF-8", to = "gbk")
# iconv(train$make, from = "UTF-8", to = "gbk")

user <- train %>% select(-label) %>% rbind(test) %>% 
  mutate(label = c(train$label, rep(-1, nrow(test))))

# Visualization
make_tr <- ggplot(data = train, mapping = aes(x = make)) + 
  geom_bar() + theme_bw()
make_te <- ggplot(data = test, mapping = aes(x = make)) + 
  geom_bar() + theme_bw()
ggarrange(make_tr, make_te, ncol = 2, nrow = 1)

model_tr <- ggplot(data = train, mapping = aes(x = model)) + 
  geom_bar() + theme_bw()
model_te <- ggplot(data = test, mapping = aes(x = model)) + 
  geom_bar() + theme_bw()
ggarrange(model_tr, model_te, ncol = 2, nrow = 1)


# glob2rx("[*")
# glob2rx("*]")
user <- user %>% mutate(tagid = str_replace(tagid, "^\\[", "") %>% 
                          str_replace("]$", "") %>% str_split(","),
                        time = str_replace(tagid, "^\\[", "") %>% 
                          str_replace("]$", "") %>% str_split(","))

## Method 1: text2Vec by R
sentences <- user$tagid

# Create iterator over tokens
# tokens<- word_tokenizer(sentences)
# 设置迭代器
it    <- itoken(sentences, progressbar = FALSE)

# 创建字典：包含唯一词及对应统计量，修剪低频词语
vocab <- create_vocabulary(it) %>% prune_vocabulary(term_count_min = 5L) #[1] 58643  3

# 设置形成语料文件
vectorizer <- vocab_vectorizer(vocab)

# 构建TCM矩阵，传入迭代器和语料文件
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)

emb_size <- 32

# glove <- GlobalVectors$new(rank = emb_size, x_max = 10)
glove <- GloVe$new(rank = emb_size, x_max = 10, learning_rate = 0.10) 
wv_main <- glove$fit_transform(tcm, n_iter = 10)
dim(wv_main) # [1] 58643    32
wv_cont <- glove$components
wv <- wv_main + t(wv_cont)


class(glove$components)
# generate embedding matrix
emb_matrix <- NULL
for (i in 1:length(sentences)) {
  
  if(i%%100 == 0) cat("Run the tagid of user:", i)
  vec <- NULL
  for (w in sentences[[i]]) {
    if (w %in% rownames(wv)) {
      vec <- cbind(vec, wv[w,])
    }
  }
  
  if (length(vec) > 0) {
    emb_matrix <- cbind(emb_matrix, rowMeans(vec, na.rm = TRUE))
  } else {
    emb_matrix <- cbind(emb_matrix, rep(0, emb_size))
  }
}


## Method 2: Word2Vec by python
# emb_tibble <- fread("./output/emb_mat.csv", header = FALSE) %>% as_tibble()

# add the embedding results into user data
for (c in 1:32) user[paste0("tagid_emb_", c)] <- emb_matrix[c,]


# label encoding 
user$gender <- as.integer(user$gender)
user$age    <- as.integer(user$age)
table(user$gender, useNA = "ifany")
table(user$age, useNA = "ifany")

user$gender <- ifelse(is.na(user$gender), 3, user$gender)
user$age    <- ifelse(is.na(user$age), 7, user$age)

sink("./output/cate_level.txt")
cat("the level of province :", "\n")
table(user$province)
cat("the level of city :", "\n")
table(user$city)
cat("the level of gender :", "\n")
table(user$gender)
cat("the level of age :", "\n")
table(user$age)
sink()

## model
inter_model <- intersect(unique(train$model), unique(test$model))
inter_make  <- intersect(unique(train$make), unique(test$make))

sum(train$model %in% inter_model) / nrow(train)
sum(test$model %in% inter_model) / nrow(test)

sum(train$make %in% inter_make) / nrow(train)
sum(test$make %in% inter_make) / nrow(test)

#将不在交集中的level编码为other
user <- user %>% 
  mutate(model = fct_collapse(model, 
    other = setdiff(unique(user$model), inter_model), 
    vivo = c("vivo", "VIVO")), 
    make = fct_collapse(make, 
      other = setdiff(unique(user$make), inter_make)))

# 找出model中的minority类别
min_model <- names(table(user$model)[table(user$model) < 100])

# 将model中的少数类别编码minority
user <- user %>% 
  mutate(model = fct_collapse(model, minority = min_model))

# 找出make中的minority类别
min_make <- names(table(user$make)[table(user$make) < 5])

# 将model中的少数类别编码minority
user <- user %>% 
  mutate(make = fct_collapse(make, minority = min_make))

ggplot(data = user) + geom_bar(mapping = aes(x = model))
ggplot(data = user) + geom_bar(mapping = aes(x = make))
