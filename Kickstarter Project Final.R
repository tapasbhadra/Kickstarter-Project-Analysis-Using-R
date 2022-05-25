#Importing the libraries
library(tidyverse)
library(caret)
library(tree)
library(class)
library(glmnet)
library(ROCR)
library(ggplot2)
library(tm)
library(text2vec)
library(text2vec)
library(SnowballC)
library(glmnet)
library(vip)

#importing the dataset
df<-read.csv('ks_training_X.csv')
train_y<-read.csv('ks_training_y.csv')
test<-read.csv('ks_test_X.csv')


# Binding X and Y
X<-rbind(df, test)

# Merge X and Y
#train<-merge(df, train_y, by='id')

# Exploratory Analysis
# Selecting features
train <-X %>%
  select(id, creator_name,goal,region, category_parent, num_words, contains_youtube, 
         sentence_counter, avgsentencelength, grade_level, afinn_pos, afinn_neg, 
         ADV, NOUN, ADP, PRT, DET, PRON, VERB, NUM, CONJ, ADJ, isTextPic, isLogoPic, isCalendarPic, isDiagramPic, isShapePic, reward_descriptions) %>%
  mutate(region = as.factor(region),
         category_parent = as.factor(category_parent),
         num_words_bin = case_when(
           num_words > 0 & num_words <=100 ~ 'Very short',
           num_words > 100 & num_words <=300 ~ 'Short',
           num_words > 300 & num_words <=750 ~ 'Long',
           num_words > 750 ~ 'Very Long',
           TRUE~'Unknown'),
         num_words_bin = as.factor(num_words_bin),
         contains_youtube = as.factor(contains_youtube),
         sentence_counter_bin = case_when(
           sentence_counter > 0 & sentence_counter <=50 ~ 'Short',
           sentence_counter > 50 & sentence_counter <=100 ~ 'Medium',
           sentence_counter > 100 ~ 'Long',
           TRUE~'Unknown'),
         sentence_counter_bin = as.factor(sentence_counter_bin),
         avgsentencelength_bin = case_when(
           avgsentencelength > 0 & avgsentencelength <=20 ~ 'Short',
           avgsentencelength > 20 & avgsentencelength <=30 ~ 'Ideal',
           avgsentencelength > 30  ~ 'Dont know English',
           TRUE~'Unknown'
         ),
         avgsentencelength_bin = as.factor(avgsentencelength_bin),
         grade_level_bin = case_when(
           grade_level < 10 ~ 'God',
           grade_level > 10  & grade_level <=20~ 'Graduate',
           grade_level > 20 ~ 'Chutiya',
           TRUE~'Unknown'
         ),
         grade_level_bin = as.factor(grade_level_bin),
         afinn_pos_bin = case_when(
           afinn_pos > 0 & afinn_pos <=20 ~ 'Low',
           afinn_pos > 20 & afinn_pos <=50 ~ 'Medium',
           afinn_pos > 50  ~ 'High',
           TRUE~'Unknown'
         ),
         afinn_pos_bin = as.factor(afinn_pos_bin),
         afinn_neg_bin = case_when(
           afinn_neg > 0 & afinn_neg <=20 ~ 'Low',
           afinn_neg > 20 & afinn_neg <=50 ~ 'Medium',
           afinn_neg > 50  ~ 'High',
           TRUE~'Unknown'
         ),
         afinn_neg_bin = as.factor(afinn_neg_bin),
         isTextPic = ifelse(is.na(isTextPic), 'Null', isTextPic),
         isTextPic = as.factor(isTextPic),
         isLogoPic = ifelse(is.na(isLogoPic), 'Null', isLogoPic),
         isLogoPic = as.factor(isLogoPic),
         isCalendarPic = ifelse(is.na(isCalendarPic), 'Null', isCalendarPic),
         isCalendarPic = as.factor(isCalendarPic),
         isDiagramPic = ifelse(is.na(isDiagramPic), 'Null', isDiagramPic),
         isDiagramPic = as.factor(isDiagramPic),
         isShapePic = ifelse(is.na(isShapePic), 'Null', isShapePic),
         isShapePic = as.factor(isShapePic)
  )

#PLotting histograms to check the distribution of the features
ggplot(train, aes(x=goal)) + geom_histogram() # We can see that the the distribution of goal is right skewed. 
ggplot(train, aes(x=ADV)) + geom_histogram() #Same with ADV
ggplot(train, aes(x=NOUN)) + geom_histogram()

#It is important to normalize these features to make sure they meet the normal distribution assumptions of the models. 
# Normalizing Goal
train$goal_norm = log(train$goal+1)
train$ADV_norm = log(train$ADV+1)
train$NOUN_norm = log(train$NOUN+1)
train$ADP_norm = log(train$ADP+1)
train$PRT_norm = log(train$PRT+1)
train$DET_norm = log(train$DET+1)
train$PRON_norm = log(train$PRON+1)
train$VERB_norm = log(train$VERB+1)
train$NUM_norm = log(train$NUM+1)
train$CONJ_norm = log(train$CONJ+1)
train$ADJ_norm = log(train$ADJ+1)

#Checking the distributions after taking log transformation
ggplot(train, aes(x=goal_norm)) + geom_histogram() # The distribution is normal
ggplot(train, aes(x=ADV_norm)) + geom_histogram()
ggplot(train, aes(x=NOUN_norm)) + geom_histogram()



names = read_csv("baby_names.csv") 
names <- names %>%
  mutate(firstname = toupper(name),
         female_name = ifelse(percent_female >= .5, 1, 0)) %>%
  select(firstname, female_name)

train <- train %>%
  separate(creator_name, into = ("firstname"), extra = "drop") %>%
  mutate(firstname = toupper(firstname)) %>%
  left_join(names, by = "firstname") %>%
  mutate(female_name = case_when(
    female_name == 1 ~ "F",
    female_name == 0 ~ "M",
    is.na(female_name) ~  "U"),
    female_name = as.factor(female_name))

#Selecting final variables
train_final <- train %>%
  select(id, female_name,goal_norm,region, category_parent, num_words_bin, contains_youtube, 
         sentence_counter_bin, avgsentencelength_bin, grade_level_bin, afinn_pos_bin, afinn_neg_bin, 
         ADV_norm, NOUN_norm, ADP_norm, PRT_norm, DET_norm, PRON_norm, VERB_norm, NUM_norm, CONJ_norm, ADJ_norm,
         isTextPic, isLogoPic, isCalendarPic, isDiagramPic, isShapePic, reward_descriptions)



# Extracting the relevant keywords from the rewards descriptions
desc_tr <- train[0:97420,] %>%
  select(id, reward_descriptions)

unlabeled_text <- train[97421:108728,] %>%
  select(id, reward_descriptions)

prep_fun = tolower
cleaning_tokenizer <- function(v) {
  v %>%
    #removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(stopwords(kind="en")) %>% #remove stopwords
    stemDocument %>%
    word_tokenizer 
}
tok_fun = cleaning_tokenizer
it_train = itoken(desc_tr$reward_descriptions, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = desc_tr$id, 
                  progressbar = FALSE)
vocab = create_vocabulary(it_train)
vocab = prune_vocabulary(vocab, term_count_min = 4500, doc_proportion_max = 0.3)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer) # A  documentterm frequency matrix constists of a lot of terms.Hence we create sparse matrices. 
dim(dtm_train)


# Converting test data
it_unlabeled = tok_fun(prep_fun(unlabeled_text$reward_descriptions))
it_unlabeled = itoken(it_unlabeled, ids = unlabeled_text$id, progressbar = FALSE)
dtm_unlabeled = create_dtm(it_unlabeled, vectorizer)
dim(dtm_unlabeled)


train_final <- train_final %>%
  select(-reward_descriptions)

# Creating dummy varaibles
dummy <- dummyVars( ~ . , data=train_final)
train_final_dummy <- data.frame(predict(dummy, newdata =train_final))


# Seperating Training and testing data
tr <- train_final_dummy[0:97420,]
test<- train_final_dummy[97421:108728,]


# Binding the text descriptions with the dataframes
dense = as.matrix(dtm_train)+0
# Use cbind() to append the columns
train_withterms <- cbind(tr, dense)
dense = as.matrix(dtm_unlabeled)+0
# Use cbind() to append the columns
test_withterms <- cbind(test, dense)


# Appending training data with the y variable
clean_tr <- train_withterms %>%
  left_join(train_y, by = 'id') %>%
  mutate(big_hit = as.factor(big_hit)) %>%
  select(-success, -backers_count)%>%
  mutate(big_hit = ifelse(big_hit=="YES",1,0))


# Split into training and validation
set.seed(1)
va_inds <- sample(nrow(clean_tr), .3*nrow(clean_tr))
train_mozi<- clean_tr[-va_inds,]
valid_mozi <- clean_tr[va_inds,]


tr_final = train_mozi
va = valid_mozi

tr_final <- tr_final %>%
  select(-success)
va <- va %>%
  select(-success)


#tr_final$big_hit <- ifelse(tr_final$big_hit=="YES",1,0)
#va$big_hit <- ifelse(va$big_hit=="YES",1,0)


# Log fitiing curves

# Using logistic regression on training and validation set
lm <- glm(big_hit~., data=tr_final, family = "binomial")
preds <- predict(lm, newdata = va, type = "response")

pred_val <- prediction(preds, va$big_hit)
performance(pred_val, measure = "auc")@y.values[[1]]

roc_log <- performance(pred_val, "tpr", "fpr")
plot(roc_log, col = "red", lwd = 2)

new <- tr_final %>%
  select(-c(goal_norm))


lm2 <- glm(big_hit~., data=new, family = "binomial")
preds <- predict(lm2, newdata = va, type = "response")

pred_val <- prediction(preds, va$big_hit)
performance(pred_val, measure = "auc")@y.values[[1]]

roc_log <- performance(pred_val, "tpr", "fpr")
plot(roc_log, col = "blue", lwd = 2, add= TRUE)

# xgboost- final

library(xgboost)
train_y <- train_y %>%
  dplyr::select(big_hit) %>%
  mutate(big_hit = ifelse(big_hit == 'NO', 0, 1))

train_withterms <- Matrix(data.matrix(train_withterms), sparse = TRUE)

set.seed(1)
train <- sample(nrow(train_withterms),.7*nrow(train_withterms))
x_train <- train_withterms[train,]
x_valid <- train_withterms[-train,]

y_train <- train_y[train,]
y_valid <- train_y[-train,]

dim(x_train)
xgboost_train = xgb.DMatrix(data=x_train, label=y_train)
xgboost_val = xgb.DMatrix(data=x_valid, label=y_valid)

watchlist = list(train = xgboost_train, test=xgboost_val)

model <- xgb.train(data = xgboost_train,max.depth=4,eta =0.01, nrounds=300, eval_metric = 'auc',
                   objective = "binary:logistic", watchlist = watchlist)




# xgboost with nrounds as hyperparamter

eta_choose <- c(100,300,500)

va_acc <- rep(0, length(eta_choose))
tr_acc <- rep(0, length(eta_choose))

for(j in c(1:length(eta_choose)))
{
  thisneta <- eta_choose[j]
  
  inner_bst <- xgb.train(data = xgboost_train,max.depth=4,eta =0.01, nrounds=thisneta, eval_metric = 'auc',
                     objective = "binary:logistic", watchlist = watchlist)
  

  inner_bst_pred_valid <- predict(inner_bst, xgboost_val, type= "response")
  pred_val <- prediction(inner_bst_pred_valid, y_valid)
  inner_bst_acc_valid<-performance(pred_val, measure = "auc")@y.values[[1]]
  
  inner_bst_pred_train <- predict(inner_bst, xgboost_train, type= "response")
  pred_val <- prediction(inner_bst_pred_train, y_train)
  inner_bst_acc_train <- performance(pred_val, measure = "auc")@y.values[[1]]
  va_acc[j] <- inner_bst_acc_valid
  tr_acc[j] <- inner_bst_acc_train
}

plot(eta_choose, va_acc, col = 'blue', xlab = 'n rounds', ylab = 'AUC', type = 'l', ylim=c(0.78,0.85))
lines(eta_choose, tr_acc, col = 'red' )
legend(x=300, y =0.8, legend = c('Training AUC',"Validation AUC"), col = c('red', 'blue'), lty = 1, cex = 0.8, text.font = 2)




# xgboost with depth as hyperparamter

eta_choose <- c(3,4,5)

va_acc <- rep(0, length(eta_choose))
tr_acc <- rep(0, length(eta_choose))

for(j in c(1:length(eta_choose)))
{
  thisneta <- eta_choose[j]
  
  inner_bst <- xgb.train(data = xgboost_train,max.depth=thisneta,eta =0.01, nrounds=200, eval_metric = 'auc',
                         objective = "binary:logistic", watchlist = watchlist)
  
  
  inner_bst_pred_valid <- predict(inner_bst, xgboost_val, type= "response")
  pred_val <- prediction(inner_bst_pred_valid, y_valid)
  inner_bst_acc_valid<-performance(pred_val, measure = "auc")@y.values[[1]]
  
  inner_bst_pred_train <- predict(inner_bst, xgboost_train, type= "response")
  pred_val <- prediction(inner_bst_pred_train, y_train)
  inner_bst_acc_train <- performance(pred_val, measure = "auc")@y.values[[1]]
  va_acc[j] <- inner_bst_acc_valid
  tr_acc[j] <- inner_bst_acc_train
}

plot(eta_choose, va_acc, col = 'blue', xlab = 'Max Depth', ylab = 'AUC', type = 'l', ylim=c(0.78,0.85))
lines(eta_choose, tr_acc, col = 'red' )
legend(x=4, y =0.8, legend = c('Training AUC',"Validation AUC"), col = c('red', 'blue'), lty = 1, cex = 0.8, text.font = 2)


# xgboost with ETA as hyperparamter

eta_choose <- c(0.01,0.03,0.05)

va_acc <- rep(0, length(eta_choose))
tr_acc <- rep(0, length(eta_choose))

for(j in c(1:length(eta_choose)))
{
  thisneta <- eta_choose[j]
  
  inner_bst <- xgb.train(data = xgboost_train,max.depth=3,eta =thisneta, nrounds=200, eval_metric = 'auc',
                         objective = "binary:logistic", watchlist = watchlist)
  
  
  inner_bst_pred_valid <- predict(inner_bst, xgboost_val, type= "response")
  pred_val <- prediction(inner_bst_pred_valid, y_valid)
  inner_bst_acc_valid<-performance(pred_val, measure = "auc")@y.values[[1]]
  
  inner_bst_pred_train <- predict(inner_bst, xgboost_train, type= "response")
  pred_val <- prediction(inner_bst_pred_train, y_train)
  inner_bst_acc_train <- performance(pred_val, measure = "auc")@y.values[[1]]
  va_acc[j] <- inner_bst_acc_valid
  tr_acc[j] <- inner_bst_acc_train
}

plot(eta_choose, va_acc, col = 'blue', xlab = 'ETA', ylab = 'AUC', type = 'l', ylim=c(0.78,0.85))
lines(eta_choose, tr_acc, col = 'red' )
legend(x=0.03, y =0.8, legend = c('Training AUC',"Validation AUC"), col = c('red', 'blue'), lty = 1, cex = 0.8, text.font = 2)



#### RF- not working
eta_choose <- c(100,200,500)

va_acc <- rep(0, length(eta_choose))
tr_acc <- rep(0, length(eta_choose))

library(randomForest)
for(j in c(1:length(eta_choose)))
{
  thisneta <- eta_choose[j]
  
  inner_bst <- randomForest(big_hit~.,
                            data=tr_final,
                            mtry=4, ntree=thisneta,
                            importance=TRUE)
  inner_bst_pred_valid <- predict(inner_bst, va, type= "response")
  pred_val <- prediction(inner_bst_pred_valid, va$big_hit)
  va_acc[j]  = performance(pred_val, measure = "auc")@y.values[[1]]
  
  inner_bst_pred_train <- predict(inner_bst, tr_final, type= "response")
  pred_val <- prediction(inner_bst_pred_train, tr_final$big_hit)
  tr_acc[j]  = performance(pred_val, measure = "auc")@y.values[[1]]
}


# GBM with depth as hyper parameter

eta_choose <- c(2,3,4)



va_acc <- rep(0, length(eta_choose))
tr_acc <- rep(0, length(eta_choose))



for(j in c(1:length(eta_choose)))
{
  thisneta <- eta_choose[j]
  
  inner_bst <- gbm(big_hit~.,data=tr_final,
                   distribution="bernoulli",
                   n.trees=200,
                   interaction.depth=thisneta)
  inner_bst_pred_valid <- predict(inner_bst, va, type= "response")
  pred_val <- prediction(inner_bst_pred_valid, va$big_hit)
  va_acc[j]  = performance(pred_val, measure = "auc")@y.values[[1]]
  
  inner_bst_pred_train <- predict(inner_bst, tr_final, type= "response")
  pred_val <- prediction(inner_bst_pred_train, tr_final$big_hit)
  tr_acc[j]  = performance(pred_val, measure = "auc")@y.values[[1]]
  va_acc[j] <- inner_bst_acc_valid
  tr_acc[j] <- inner_bst_acc_train
}

plot(eta_choose, va_acc, col = 'blue', xlab = 'Depth', ylab = 'AUC', type = 'l')
lines(eta_choose, tr_acc, col = 'red' )
legend(x=3, y =0.7, legend = c('Training AUC',"Validation AUC"), col = c('red', 'blue'), lty = 1, cex = 0.8, text.font = 2)
