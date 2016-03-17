
setwd("C:/Kaggle/BNP/")

rm(list = ls())

require(xgboost)
require(caret)

`%|%` <- function(a,b) paste0(a,b)

#constants
data_set <- 'ds01'
out_fdr <- '6_Results'
in_fdr <- '2_Data_Use'
ft_fdr <- '3_features'
ft_set <- 'fs01'
version <- 'V4'
filename <- 'xgb_' %|% data_set %|% '_' %|% ft_set %|% '_' %|% version

#------------------------------------------------------------------------

#Specify Dataset - Dataset 1
#Load in dataset:
load(in_fdr %|% '/' %|% data_set %|% '/train_' %|% data_set %|% '.RData')
load(in_fdr %|% '/' %|% data_set %|% '/test_' %|% data_set %|% '.RData')
load(in_fdr %|% '/' %|% data_set %|% '/holdout_' %|% data_set %|% '.RData')

set.seed(123)

num_groups <- length(train)
num_folds  <- 8

#------------------------------------------------------------------------

#Specify Feature Selection
#Featureset 1 - iTranslates factor variables into numeric
source(ft_fdr %|% '/' %|% ft_set %|% '.R')
train <- factor_to_numeric(train)
test <- factor_to_numeric(test)
holdout <- factor_to_numeric(holdout)

train <- replace_na_with_fixed_val(train, v=-1)
test <- replace_na_with_fixed_val(test, v=-1)
holdout <- replace_na_with_fixed_val(holdout, v=-1)

#------------------------------------------------------------------------

#Split each set into 8 cross folds
folds <- list()
for (g in 1:num_groups){
    folds[[g]] <- createFolds(train[[g]]$target, num_folds)  
}

#copy independent variable to separate list
y_train <- list()
for (i in 1:length(train)){
    y_train[[i]] <- train[[i]][, "target"]
}

y_cv <- list()
for (i in 1:length(train)){
    y_cv[[i]] = list()
    for (f in 1:num_folds){
        y_cv[[i]][[f]] = list()
        y_cv[[i]][[f]] = y_train[[i]][folds[[i]][[f]]]
    }
}

y_holdout <- list()
for (i in 1:length(holdout)){
    y_holdout[[i]] <- holdout[[i]][, "target"]
}

#Remove group and target columns
for (i in 1:length(train)){
    train[[i]] <- train[[i]][, setdiff(names(train[[i]]),c("group","target"))]
}

for (i in 1:length(test)){
    test[[i]] <- test[[i]][, setdiff(names(test[[i]]),"group")]
}

for (i in 1:length(holdout)){
    holdout[[i]] <- holdout[[i]][, setdiff(names(holdout[[i]]),c("group","target"))]
}

#------------------------------------------------------------------------

#Specify model Selection

#Define parameters for the model
# grid <- expand.grid(eta=c(0.03,0.1,0.3,1),gamma=c(0,0.03,1,3))
# grid$LL <- NA
# 
# for (i in 1:nrow(grid)){
#     ..refer to factors
#     grid$gamma[i]
#     add output to grid
#     grid$LL[i] <- result
#     
# }

MultiLogLoss <- function(act, pred)
{
    eps = 1e-15;
    nr <- nrow(pred)
    pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
    pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
    ll = sum(act*log(pred) + (1-act)*log(1-pred))
    ll = ll * -1/(nrow(act))      
    return(ll);
}

eta = c(0.01)
max_depth = c(10)
gamma = c(0)
num_trees = c(110, 120, 130)

if (file.exists(out_fdr %|% '/' %|% filename %|% '_cv_tuning.csv')){
    LL_cv_test <- read.csv(out_fdr %|% '/' %|% filename %|% '_cv_tuning.csv')
} else {
    LL_cv_test = data.frame()
}

for (e in eta){
    for (d in max_depth) {
        for (n in num_trees) {
            for (g in gamma) {
                
                mdl <- 'xgb_' %|% data_set %|% '_' %|%
                    ft_set %|% '_' %|% version %|% '_eta_' %|% e %|% '_depth_' %|% d %|%
                    '_trees_' %|% n %|% '_gamma_' %|% g
                
                param0 <- list(
                    # general , non specific parameters
                    "objective"  = "binary:logistic"
                    , "eval_metric" = "logloss"
                    , "eta" = e
                    , "subsample" = 1
                    , "colsample_bytree" = 0.3
                    , "min_child_weight" = 1
                    , "max_depth" = d
                    , "gamma" = g
                )
                
                #Define XGB Model
                doTest <- function(y_train, train, test, param0, iter) {
                    n<- nrow(train)
                    xgtrain <- xgb.DMatrix(as.matrix(train), label = y_train)
                    
                    xgval_train = xgb.DMatrix(as.matrix(train))
                    xgval_test = xgb.DMatrix(as.matrix(test))    
                    
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
                    
                    rm(model)
                    p = list()
                    p[[1]] <- pred_test
                    p[[2]] <- pred_train
                    p
                }
                
                #Fit Model
                
                PredictedProb_cv = list()
                PredictedProb_train = list()
                
                for (i in 1:length(train)){
                    PredictedProb_cv[[i]] = list()
                    PredictedProb_train[[i]] = list()
                    for (f in 1:num_folds){
                        PredictedProb_cv[[i]][[f]] = list()
                        PredictedProb_train[[i]][[f]] = list()
                    }
                }
                
                
                for (i in 1:length(train)){
                    for (f in 1:num_folds){    
                        predictions <- doTest(y_train[[i]][-folds[[i]][[f]]], 
                                              train[[i]][-folds[[i]][[f]], ], 
                                              train[[i]][folds[[i]][[f]], ], 
                                              param0, n)
                        
                        PredictedProb_cv[[i]][[f]] <- predictions[[1]]
                        PredictedProb_train[[i]][[f]] <- predictions[[2]]
                        rm(predictions)
                    }
                }
                
                LL_cv <- list()
                
                for (i in 1:length(train)){
                    LL_cv[[i]] = list()
                    for (f in 1:num_folds){
                        LL_cv[[i]][[f]] = list()
                        LL_cv[[i]][[f]] = MultiLogLoss(data.frame(y_cv[[i]][[f]]), 
                                                       data.frame(PredictedProb_cv[[i]][[f]]))
                    }
                }
                
                LL_cv_df = data.frame()
                
                for (i in 1:length(train)){
                    LL_cv_df[i, "name"] = mdl
                    LL_cv_df[i, "mean"] = mean(unlist(LL_cv[[i]]))
                    LL_cv_df[i, "sd"] = sd(unlist(LL_cv[[i]]))
                }
                
                LL_cv_test = rbind(LL_cv_test, LL_cv_df)
                
            }
            
        }
        
    }
    
    write.csv(LL_cv_test, out_fdr %|% '/' %|% filename %|% '_cv_tuning.csv', row.names = F)
    
}

LL_cv_test



# #Create train prediction data
# submissionlist_train <- list()
# for (i in 1:length(train)){
#     submissionlist_train[[i]] <- data.frame("ID" = train[[i]][, 1], 
#                                             "PredictedProb" = PredictedProb_train[[i]])
# }
# 
# submission_train <- do.call("rbind", submissionlist_train)
# submission_train <- submission_train[with(submission_train, order(ID)), ]
# 
# write.csv(submission_train, out_fdr %|% '/XGB_' %|% data_set %|% '_' %|% ft_set %|% '_' %|% version %|% '_train.csv', row.names=F, quote=F)
# 
# 
# #Create holdout prediction data
# submissionlist_holdout <- list()
# for (i in 1:length(holdout)){
#     submissionlist_holdout[[i]] <- data.frame("ID" = holdout[[i]][, 1], 
#                                               "PredictedProb" = PredictedProb_holdout[[i]])
# }
# 
# submission_holdout <- do.call("rbind", submissionlist_holdout)
# submission_holdout <- submission_holdout[with(submission_holdout, order(ID)), ]
# 
# write.csv(submission_holdout, out_fdr %|% '/XGB_' %|% data_set %|% '_' %|% ft_set %|% '_' %|% version %|% '_holdout.csv', row.names=F, quote=F)
# 
# 
# #Create submission data
# submissionlist_test <- list()
# for (i in 1:length(test)){
#     submissionlist_test[[i]] <- data.frame("ID" = test[[i]][, 1], 
#                                            "PredictedProb" = PredictedProb_test[[i]])
# }
# 
# submission_test <- do.call("rbind", submissionlist_test)
# submission_test <- submission_test[with(submission_test, order(ID)), ]
# 
# write.csv(submission_test, out_fdr %|% '/XGB_' %|% data_set %|% '_' %|% ft_set %|% '_' %|% version %|% '_test.csv', row.names=F, quote=F)
