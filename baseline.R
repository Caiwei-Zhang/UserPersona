
# splitting train data and test data
user_train <- user %>% filter(label != -1) %>% 
  select(-all_of(c("pid", "tagid", "time", "make")))
user_test <- user %>% filter(label == -1) %>% 
  select(-all_of(c("pid", "tagid", "time", "make", "label")))
cate_feas <- c("gender", "age", "province", "city", "model")

x_train <- user_train %>% select(-label)
y_train <- user_train$label

set.seed(2021)
train_list <- createFolds(user_train$label, k = 5, returnTrain = TRUE)

oof_train_pred <- numeric(nrow(x_train))
pred_test <- NULL

for (k in 1:5) {
  
  print(paste("Running fold:", k))
  
  cv_train_idx <- train_list[[k]]
  cv_valid_idx <- setdiff(1:nrow(x_train), cv_train_idx)
  
  cv_x_train <- x_train[cv_train_idx, ]
  cv_x_valid <- x_train[cv_valid_idx, ]
  cv_y_train <- y_train[cv_train_idx]
  cv_y_valid <- y_train[cv_valid_idx]
  
  cv_dtrain <- lgb.Dataset(data = as.matrix(cv_x_train), label = cv_y_train)
  cv_dvalid <- lgb.Dataset(data = as.matrix(cv_x_valid), label = cv_y_valid)
  
  param  <- list(objective = "binary", boosting = "gbdt", 
                 learning_rate = 0.05, eval = "auc", 
                 feature_fraction = 0.8, bagging_fraction = 0.8, 
                 bagging_freq = 3) # , max_depth = 6, num_leaves = 60, 
                 #min_child_weight = 1)
  
  clf  <- lgb.train(data = cv_dtrain, params = param, nrounds = 10000, 
                    categorical_feature = cate_feas, 
                    early_stopping_rounds = 50, eval_freq = 100,
                    valids = list(valid = cv_dvalid), force_row_wise = TRUE)
  
  oof_train_pred[cv_valid_idx] <- predict(clf, data = as.matrix(cv_x_valid), )
  tmp_test_pred  <- predict(clf, data = as.matrix(user_test))
  
  pred_test <- cbind(pred_test, tmp_test_pred)
}  
  

pred_test_mean <- rowMeans(pred_test)

# evaluation
auc <- roc.curve(y_train, oof_train_pred)$auc
sprintf("AUC score: %f", auc)

f1_score <- 2 * accuracy.meas(response = y_train, predicted = oof_train_pred)$F
sprintf("F1 score: %f", f1_score)


# save the results
submit <- data.frame(user_id = test$pid, category_id = ifelse(pred_test_mean >= 0.5, 1, 0))
filename <- paste0(round(f1_score, 5), ".csv")

write.csv(submit, paste0("./res/", filename), row.names = FALSE)







# Tuning parameter
# define a eval function
lgb_f1_score <- function(preds, dtrain) {
  
  f1_score <- 2 * accuracy.meas(response = dtrain$getinfo("label"), 
                                predicted = preds)$F
  res <- list(name = "f1_score", 
              value = ifelse(is.nan(f1_score), 0, f1_score),
              higher_better = TRUE)
  return(res)
}


### Cross-validation
tr_va <- createDataPartition(user_train$label, times = 1, p = 0.9, list = FALSE)
x.train <- x_train[tr_va, ]
y.train <- user_train$label[tr_va]

x.valid <- x_train[-tr_va, ]
y.valid <- user_train$label[-tr_va]

## Stage 1: tune complexity of tree
# 1. max_depth, num_leaves
param_grid <- expand_grid(max_depth = c(6, 8, 10), num_leaves = c(2^6, 2^7, 2^8)) 
metric_df_1 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = param_grid$max_depth[i], 
                              num_leaves = param_grid$num_leaves[i]),
                data = dtrain, nrounds = 5000, nfold = 5, stratified = TRUE, 
                early_stopping_rounds = 50, eval = lgb_f1_score, eval_freq = 100, 
                categorical_feature = cate_feas, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_score$eval)),
              clf$record_evals$valid$f1_score$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_score$eval))]])
  
  metric_df_1 <- rbind(metric_df_1, metric)
  colnames(metric_df_1) <- c("valid_eval", "valid_err")
  
}

cat("best max_depth:", param_grid$max_depth[which.max(metric_df_1$valid_eval)], "\n",
    "best num_leaves:", param_grid$num_leaves[which.max(metric_df_1$valid_eval)], "\n")



# 2. details of num_leaves
param_grid <- expand_grid(num_leaves = seq(50, 70, 10)) 
metric_df_2 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = 6, num_leaves = param_grid$num_leaves[i]),
                data = dtrain, nrounds = 5000, nfold = 5, stratified = TRUE, 
                early_stopping_rounds = 50, eval = lgb_f1_score, eval_freq = 100, 
                categorical_feature = cate_feas, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_score$eval)),
              clf$record_evals$valid$f1_score$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_score$eval))]])
  
  metric_df_2 <- rbind(metric_df_2, metric)
  colnames(metric_df_2) <- c("valid_eval", "valid_err")
  
}

cat("When max_depth =", 6, "best num_leaves is:", param_grid$num_leaves[which.max(metric_df_2$valid_eval)], "\n")


## 3. min_child_weight, min_data_in_leaf
param_grid <- expand_grid(min_child_weight = seq(1, 10, 2)) 
metric_df_3 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = 6, num_leaves = 60, min_child_weight = param_grid$min_child_weight[i]),
                data = dtrain, nrounds = 5000, nfold = 5, stratified = TRUE, 
                early_stopping_rounds = 50, eval = lgb_f1_score, eval_freq = 100, 
                categorical_feature = cate_feas, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_score$eval)),
              clf$record_evals$valid$f1_score$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_score$eval))]])
  
  metric_df_3 <- rbind(metric_df_3, metric)
  colnames(metric_df_3) <- c("valid_eval", "valid_err")
  
}

cat("best min_child_weight:", param_grid$min_child_weight[which.max(metric_df_3$valid_eval)], "\n")



## 4. details of min_child_weight
param_grid <- expand_grid(min_child_weight = seq(7, 10, 1)) 
metric_df_4 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = 6, num_leaves = 60, min_child_weight = param_grid$min_child_weight[i]),
                data = dtrain, nrounds = 5000, nfold = 5, stratified = TRUE, 
                early_stopping_rounds = 50, eval = lgb_f1_score, eval_freq = 100, 
                categorical_feature = cate_feas, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_score$eval)),
              clf$record_evals$valid$f1_score$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_score$eval))]])
  
  metric_df_4 <- rbind(metric_df_4, metric)
  colnames(metric_df_4) <- c("valid_eval", "valid_err")
  
}

cat("best min_child_weight:", param_grid$min_child_weight[which.max(metric_df_4$valid_eval)], "\n")




### the best combination of parameter 
dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
param <- list(objective = "binary", boosting = "gbdt", 
              learning_rate = 0.01, 
              feature_fraction = 0.7, bagging_fraction = 0.9, 
              bagging_freq = 5, max_depth = 7, num_leaves = 100, 
              min_child_weight = 8)

opt.clf.2 <- lgb.train(data = dtrain, params = param, 
                       nrounds = 1000, categorical_feature = cate_feas, 
                       verbose = -1, force_row_wise = TRUE)


pred_valid <- predict(opt.clf.2,  data = as.matrix(x_valid[feas]))
Meas_valid <- f1_score(y_pred = ifelse(pred_valid > 0.244, 1, 0), 
                       y_true = y_valid, pattern = "macro") # 0.5862859



# save submit
pred_test <- predict(opt.clf.2, data = as.matrix(car_test[feas]))
submit <- data.frame(customer_id = test$customer_id, loan_default = ifelse(pred_test > 0.25, 1, 0))
write.csv(submit, file = "./res/res19_opt.csv", row.names = FALSE)



