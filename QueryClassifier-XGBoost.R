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
dtest <- xgb.DMatrix(data.matrix(test))
dtrain <- xgb.DMatrix(data.matrix(train), 
                      label = data.matrix(trainlabelsfactored))


start.time <- Sys.time()
xgb_params = list(colsample_bytree= 0.7,
                  subsample = 0.7,
                  eta = 0.05,
                  objective= 'multi:softmax',
                  max_depth= 5,
                  min_child_weight= 1,
                  eval_metric= "mlogloss", num_class = categoryclassnos,
                  nthread=4)

fit.xgb = xgb.train(params = xgb_params,
                    data = dtrain,
                    nrounds = 500,
                    watchlist = list(train = dtrain, test=dtest),
                    print_every_n = 50)

end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken


my_preds <- predict(fit.xgb, dtest, reshape = TRUE)

# Check accuracy on training.
mat <- confusionMatrix(predicted.xgb, testdeptt



