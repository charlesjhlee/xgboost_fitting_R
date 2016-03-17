
setwd("C:/Kaggle/BNP/")

#constants
data_set <- 'ds01'
out_fdr <- '6_Results'
in_fdr <- '2_Data_Use'
ft_fdr <- '3_features'
ft_set <- 'fs01'
version <- 'V5'

`%|%` <- function(a,b) paste0(a,b)

require(xgboost)

#Specify Dataset - Dataset 1
#Load in dataset:
load(in_fdr %|% '/' %|% data_set %|% '/train_knn_' %|% data_set %|% '.RData')
load(in_fdr %|% '/' %|% data_set %|% '/test_knn_' %|% data_set %|% '.RData')
load(in_fdr %|% '/' %|% data_set %|% '/holdout_knn_' %|% data_set %|% '.RData')

train.list <- train
test.list <- test
holdout.list <- holdout

rm(train)
rm(test)
rm(holdout)

#Combine training and test datasets into one list
#train.list <- list(train.A,train.B,train.C,train.D,train.E,train.F)
#test.list <- list(test.A,test.B,test.C,test.D,test.E,test.F)

#After setting up train.list, remove individual data parts to free up memory
#rm(list=ls(pattern='^train\\..$'))
#rm(list=ls(pattern='^test\\..$'))

#Specify Feature Selection
#Featureset 1 - iTranslates factor variables into numeric
source(ft_fdr %|% '/' %|% ft_set %|% '.R')
train.list <- factor_to_numeric(train.list)
test.list <- factor_to_numeric(test.list)
holdout.list <- factor_to_numeric(holdout.list)

train.list <- replace_na_with_fixed_val(train.list, v=-1)
test.list <- replace_na_with_fixed_val(test.list, v=-1)
holdout.list <- replace_na_with_fixed_val(holdout.list, v=-1)

#Specify model Selection
#XGB Model Version 1
#Model from Kaggle forum. Let's see how well it does on the subdivided data
#with knn imputed values

#Define parameters for the model
param0 <- list(
    # general , non specific parameters
    "objective"  = "binary:logistic"
    , "eval_metric" = "logloss"
    , "eta" = 0.01
    , "subsample" = 0.8
    , "colsample_bytree" = 0.8
    , "min_child_weight" = 1
    , "max_depth" = 10
)

param0 <- list(
    # general , non specific parameters
    "objective"  = "binary:logistic"
    , "eval_metric" = "logloss"
    , "eta" = 0.01
    , "subsample" = 1
    , "colsample_bytree" = 0.3
    , "min_child_weight" = 1
    , "max_depth" = 10
    , "gamma" = 0
)

#Define XGB Model
doTest <- function(y_train, train, test, holdout, param0, iter) {
    n<- nrow(train)
    xgtrain <- xgb.DMatrix(as.matrix(train), label = y_train)
    
    xgval_train = xgb.DMatrix(as.matrix(train))
    xgval_test = xgb.DMatrix(as.matrix(test))    
    xgval_holdout = xgb.DMatrix(as.matrix(holdout))
    
    watchlist <- list('train' = xgtrain)
    model = xgb.train(
        nrounds = iter
        , params = param0
        , data = xgtrain
        , watchlist = watchlist
        , print.every.n = 100
        , nthread = 8 
    )
    
    pred_test <- predict(model, xgval_test)
    pred_train <- predict(model, xgval_train)
    pred_holdout <- predict(model, xgval_holdout)
    
    rm(model)
    gc()
    p = list()
    p[[1]] <- pred_test
    p[[2]] <- pred_train
    p[[3]] <- pred_holdout
    p
}

#copy independent variable to separate list
y_train <- list()
for (i in 1:length(train.list)){
    y_train[[i]] <- train.list[[i]][, "target"]
}

y_holdout <- list()
for (i in 1:length(holdout.list)){
    y_holdout[[i]] <- holdout.list[[i]][, "target"]
}

#Remove group and target columns
for (i in 1:length(train.list)){
    train.list[[i]] <- train.list[[i]][, setdiff(names(train.list[[i]]),c("group","target"))]
}

for (i in 1:length(test.list)){
    test.list[[i]] <- test.list[[i]][, setdiff(names(test.list[[i]]),"group")]
}

for (i in 1:length(holdout.list)){
        holdout.list[[i]] <- holdout.list[[i]][, setdiff(names(holdout.list[[i]]),c("group","target"))]
}


#Fit Model
ensemble_test <- list()
ensemble_train <- list()
ensemble_holdout <- list()
p_test <- list()
p_train <- list()
p_holdout <- list()
PredictedProb_test <- list()
PredictedProb_train <- list()
PredictedProb_holdout <- list()

for (i in 1:length(train.list)){
    ensemble_test[[i]] <- rep(0, nrow(test.list[[i]]))
    ensemble_train[[i]] <- rep(0, nrow(train.list[[i]]))
    ensemble_holdout[[i]] <- rep(0, nrow(holdout.list[[i]]))
    
    for (k in 1:5){

        predictions <- doTest(y_train[[i]], train.list[[i]], test.list[[i]], holdout.list[[i]], param0, 110)
        
        p_test[[i]] <- predictions[[1]]
        p_train[[i]] <- predictions[[2]]
        p_holdout[[i]] <- predictions[[3]]
        
        ensemble_test[[i]] <- ensemble_test[[i]] + p_test[[i]]
        ensemble_train[[i]] <- ensemble_train[[i]] + p_train[[i]]
        ensemble_holdout[[i]] <- ensemble_holdout[[i]] + p_holdout[[i]]
    }
    PredictedProb_test[[i]] <- ensemble_test[[i]]/k
    PredictedProb_train[[i]] <- ensemble_train[[i]]/k
    PredictedProb_holdout[[i]] <- ensemble_holdout[[i]]/k
}

#Create train prediction data
submissionlist_train <- list()
for (i in 1:length(train.list)){
        submissionlist_train[[i]] <- data.frame("ID" = train.list[[i]][, 1], 
                                          "PredictedProb" = PredictedProb_train[[i]])
}

submission_train <- do.call("rbind", submissionlist_train)
submission_train <- submission_train[with(submission_train, order(ID)), ]

write.csv(submission_train, out_fdr %|% '/XGB_' %|% data_set %|% '_' %|% ft_set %|% '_' %|% version %|% '_train.csv', row.names=F, quote=F)


#Create holdout prediction data
submissionlist_holdout <- list()
for (i in 1:length(holdout.list)){
        submissionlist_holdout[[i]] <- data.frame("ID" = holdout.list[[i]][, 1], 
                                          "PredictedProb" = PredictedProb_holdout[[i]])
}

submission_holdout <- do.call("rbind", submissionlist_holdout)
submission_holdout <- submission_holdout[with(submission_holdout, order(ID)), ]

write.csv(submission_holdout, out_fdr %|% '/XGB_' %|% data_set %|% '_' %|% ft_set %|% '_' %|% version %|% '_holdout.csv', row.names=F, quote=F)


#Create submission data
submissionlist_test <- list()
for (i in 1:length(test.list)){
    submissionlist_test[[i]] <- data.frame("ID" = test.list[[i]][, 1], 
                                      "PredictedProb" = PredictedProb_test[[i]])
}

submission_test <- do.call("rbind", submissionlist_test)
submission_test <- submission_test[with(submission_test, order(ID)), ]

write.csv(submission_test, out_fdr %|% '/XGB_' %|% data_set %|% '_' %|% ft_set %|% '_' %|% version %|% '_test.csv', row.names=F, quote=F)
