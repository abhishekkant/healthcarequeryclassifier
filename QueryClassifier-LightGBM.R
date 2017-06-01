library(caret)
library(dplyr)
library(xgboost)
library(snowfall)
sfInit(parallel=TRUE, cpus=2)

set.seed(701)

traintest <- read.csv("C:\\data\\Learning R\\medicalqueryDTM.csv")

# Split Data into Train and Test
trainindex <- createDataPartition(traintest$primarydeptt, p=0.75,list = FALSE)

train <- traintest[trainindex,]
test <- traintest[-trainindex,]

testdeptt <- test$primarydeptt
test$primarydeptt <- NULL


# Prepare XGBoost Model
categoryclassnos <- nlevels(train$primarydeptt)
trainlabelsfactored <- as.integer(train$primarydeptt) - 1
train$primarydeptt <- NULL
test[] <- lapply(test,as.numeric)
train[] <- lapply(train,as.numeric)


# Prepare lightgbm model
lgbtrain <- lgb.Dataset(data=data.matrix(train), label=data.matrix(trainlabelsfactored), 
                        is_sparse=TRUE,colnames= as.list(colnames(train)))
lgbtest <- lgb.Dataset.create.valid(lgbtrain, data = data.matrix(test))
valids <- list(train = lgbtrain,test=lgbtest)


start.time <- Sys.time()

params <- list(objective="multiclass",
               num_class = categoryclassnos,
               metric = "multi_logloss",
               verbose = 1, 
               num_threads = 2)


lgb_model <- lgb.train(params, 
                       data = lgbtrain, 
                       nrounds = 500,
                       num_leaves = 50,
                       valids = valids,
                       min_data=1, 
                       learning_rate=0.01,
                       early_stopping_rounds=10)


end.time <- Sys.time()
time.taken.lgbm <- round(end.time - start.time,2)
time.taken.lgbm

my_preds <- predict(fit.xgb, dtest, reshape = TRUE)

# Check accuracy on training.
mat <- confusionMatrix(predicted.xgb, testdeptt)