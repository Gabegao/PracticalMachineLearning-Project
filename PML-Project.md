# Predicting People's Exercise Manner by Using Practical Machine Learning
## by Gabegao

*Summary:*
The goal of this study is to predict the manner in which a person did his/her exercise, i.e. how well a person does the exercise. The prediction model is derived based on data from accelerometers on the belt, forearm, arm and dumbell of 6 participants. Due to the quality of the data, some filtering/cleaning steps are conducted. The training data are also partitioned to conduct cross validation. Finally, the prediction model is applied to test data and the results are resonable.

1. Loading data

```r
train<-read.csv("pml-training.csv", header=TRUE, sep=",")
test<-read.csv("pml-testing.csv", header=TRUE, sep=",")
names(train)
head(train)
```
Based there are 160 variables and lots of them have NA values, so our next step is to remove those NAs. Also, we need to remove the variables with near zero variance.

2. Cleaning data

```r
# Remove variables with NA values
temp<-colSums(is.na(train))
table(temp)
```

```
## temp
##     0 19216 
##    93    67
```

```r
# Keep only the 93 variables with non-NA values
train2 <- train[, temp==0]
# Remove variables with near zero variance
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
NZV <- nearZeroVar(train2, saveMetrics=TRUE)
Var<- names(train2)[NZV$nzv==FALSE]
train2 <- subset(train2, select=Var)
# Remove other irrelevant variables:
# X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, num_window
train2 <- train2[,-c(1:6)]
```
After the cleaning and filtering, there are 53 meaningful variables remained and the filtered data set is named as "train2".

3. Practical Machine Learning model

```r
library(caret)
# Partition data for cross validation
inTrain<- createDataPartition(y=train2$classe, p=0.75, list=FALSE)
training<- train2[inTrain,]
testing<- train2[-inTrain,]
dim(training)
dim(testing)
# Try a couple of machine learning models
set.seed(1234)
mod1<- train(classe~., method="rpart", data=training)
```

```
## Loading required package: rpart
```

```r
mod2<- train(classe~., method="rf", data=training)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
mod3<- train(classe~., method="lda", data=training)
```

```
## Loading required package: MASS
```

```r
mod4<- train(classe~., method="gbm", data=training)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```r
pred1<-predict(mod1, testing)
pred2<-predict(mod2, testing)
pred3<-predict(mod3, testing)
pred4<-predict(mod4, testing)
# Compare models accuracy and select the best one
confusionMatrix(testing$classe, pred1)$overall['Accuracy']
confusionMatrix(testing$classe, pred2)$overall['Accuracy']
confusionMatrix(testing$classe, pred3)$overall['Accuracy']
confusionMatrix(testing$classe, pred4)$overall['Accuracy']
```
Based on the results, the random forest model has the highest accuracy rate 0.9942904, following by the gradient boosting model (accuracy rate=0.9679853), then Latent Dirichlet Allocation (accuracy rate=0.7165579) and finally the rpartition model(accuracy rate=0.4930669). So we will select the random forest model as our final prediction model.

4. Apply selected model to test data

```r
confusionMatrix(testing$classe, pred2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1392    2    0    0    1
##          B    3  943    3    0    0
##          C    0    2  850    3    0
##          D    0    0    5  797    2
##          E    0    2    2    4  893
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9941         
##                  95% CI : (0.9915, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9925         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9937   0.9884   0.9913   0.9967
## Specificity            0.9991   0.9985   0.9988   0.9983   0.9980
## Pos Pred Value         0.9978   0.9937   0.9942   0.9913   0.9911
## Neg Pred Value         0.9991   0.9985   0.9975   0.9983   0.9993
## Prevalence             0.2845   0.1935   0.1754   0.1639   0.1827
## Detection Rate         0.2838   0.1923   0.1733   0.1625   0.1821
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9985   0.9961   0.9936   0.9948   0.9973
```

```r
result<- predict(mod2, newdata=test)
data.frame(test$user_name, result)
```

```
##    test.user_name result
## 1           pedro      B
## 2          jeremy      A
## 3          jeremy      B
## 4          adelmo      A
## 5          eurico      A
## 6          jeremy      E
## 7          jeremy      D
## 8          jeremy      B
## 9        carlitos      A
## 10        charles      A
## 11       carlitos      B
## 12         jeremy      C
## 13         eurico      B
## 14         jeremy      A
## 15         jeremy      E
## 16         eurico      E
## 17          pedro      A
## 18       carlitos      B
## 19          pedro      B
## 20         eurico      B
```

```r
table(result)
```

```
## result
## A B C D E 
## 7 8 1 1 3
```
