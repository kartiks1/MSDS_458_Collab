
# https://sp7091.medium.com/regression-approaches-to-predict-diamond-price-258478a485c9
# ref paper
# https://rpubs.com/Davo2812/1102821
# https://www.kaggle.com/datasets/shivam2503/diamonds
# https://rpubs.com/gokusun327/diamonddatatest
# total depth percentage = z / mean(x, y) = 2 * z / (x + y) 

setwd("C:/NorthWestern/MiscNew")

require(dplyr)
require(plyr)
require(ggplot2)
library(tidyverse) 

#df <- read.csv(file="diamondprice.csv", head=TRUE, sep=",")

data(diamonds, package = "ggplot2")
df <- diamonds

str(df)
summary(df)
df$volume <- df$x*df$y*df$z

df$class <- ifelse(df$price > 5000,"high", "low")

table(df$class)

dfsub <- df[,c("carat", "cut", "color", "clarity", "table", "depth", "volume",
               "price", "class")]



################### EDA ###########################
################# unidimensional histograms and boxplots#################################################

ggplot(df, aes(x=carat)) + 
  geom_histogram(color="black") +
  labs(title="Distribution") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df, aes(x=depth)) + 
  geom_histogram(color="black") +
  labs(title="Distribution") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df, aes(x=table)) + 
  geom_histogram(color="black") +
  labs(title="Distribution") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df, aes(x=price)) + 
  geom_histogram(color="black") +
  labs(title="Distribution") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df, aes(x=volume)) + 
  geom_histogram(color="black") +
  labs(title="Distribution") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df, aes(x=reorder(cut,price), y=price)) + 
  geom_boxplot(fill="gray")+
  labs(title="Distribution of Price")+
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df, aes(x=reorder(color,price), y=price)) + 
  geom_boxplot(fill="gray")+
  labs(title="Distribution of Price")+
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df, aes(x=reorder(clarity,price), y=price)) + 
  geom_boxplot(fill="gray")+
  labs(title="Distribution of Price")+
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

###################### tables ############################
tab1 <- as.data.frame(table(df$cut))
tab2 <- as.data.frame(table(df$color))
tab3 <- as.data.frame(table(df$clarity))
tab4 <- as.data.frame(table(df$cut, df$color, df$clarity))

library(doBy)
tab5 <- as.data.frame(summaryBy(price + carat ~ cut+color+clarity, data = df, 
          FUN = function(x) { c(m = mean(x), s = sd(x), n= length(x)) } ) )


###############################  scatter plots  ##################################################

ggplot(df, aes(x=carat, y=price)) + geom_point(size=3) +
  ggtitle("Scatterplot") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5)) +
  geom_smooth(method=lm, se=FALSE)

ggplot(df, aes(x=table, y=price)) + geom_point(size=3) +
  ggtitle("Scatterplot") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5)) +
  geom_smooth(method=lm, se=FALSE)

ggplot(df, aes(x=depth, y=price)) + geom_point(size=3) +
  ggtitle("Scatterplot") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5)) +
  geom_smooth(method=lm, se=FALSE)

ggplot(df, aes(x=volume, y=price)) + geom_point(size=3) +
  ggtitle("Scatterplot") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5)) +
  geom_smooth(method=lm, se=FALSE)

ggplot(df, aes(x=carat, y=price, shape = factor(cut), color=color)) + geom_point(size=3) +
  ggtitle("Scatterplot") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5)) +
  geom_smooth(method=lm, se=FALSE)

ggplot(df, aes(x=carat, y=price, shape = factor(clarity), color=color)) + geom_point(size=3) +
  ggtitle("Scatterplot") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5)) +
  geom_smooth(method=lm, se=FALSE)

########################  Box plot with 2 factors #########################


ggplot(df, aes(x=factor(clarity), y=price, fill=factor(color))) + 
  geom_boxplot()+
  labs(title="Distribution")+
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#################### SMOTE ######################

####################### smote #############################
library(smotefamily) # only numeric

library(themis) ## both categorical & numeric
library(caret)


#install.packages("devtools")
#devtools::install_github("dongyuanwu/RSBID")
dfsub$class <- as.factor(dfsub$class)
newdata <- smotenc(dfsub, var="class", k = 5, over_ratio = 0.7) # requires themis

table(newdata$class)

################# Random Forest #########################

require(rpart)
require(rpart.plot)
require(tree)
require(rattle)
require(caTools)
require(ROCR)
require(ResourceSelection)
library(corrgram)
library(MASS)
library(randomForest)
library(inTrees)
library(pROC)
library(caret)
library(dplyr)
library("e1071")

rf1 <- randomForest(price ~ carat + cut + color + clarity + depth + table + x+y+z, 
                    data=df,importance=TRUE, ntree=100)

rf1smote <- randomForest(price ~ carat + cut + color + clarity + + table + depth + volume, 
                    data=newdata,importance=TRUE, ntree=100)
print(rf1)
print(rf1smote)

plot(rf1)
plot(rf1smote)

importance(rf1)
varImpPlot(rf1)

importance(rf1smote)
varImpPlot(rf1smote)

y <- rf1$y
p <- rf1$predicted

df2 <- data.frame(y,p)
ggplot(df2, aes(x = y, y = p)) +
  geom_point() +  # Add points
  geom_abline(slope = 1, intercept = 0) + 
  ggtitle("Scatterplot - Predicted vs Actual price") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

df2$err <- abs(df2$y-df2$p)
df2$perr <- df2$err/df2$y
mean(df2$err)
mean(df2$perr)

y <- rf1smote$y
p <- rf1smote$predicted

df2 <- data.frame(y,p)
ggplot(df2, aes(x = y, y = p)) +
  geom_point() +  # Add points
  geom_abline(slope = 1, intercept = 0) + 
  ggtitle("Scatterplot - Predicted vs Actual price") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

df2$err <- abs(df2$y-df2$p)
df2$perr <- df2$err/df2$y
mean(df2$err)
mean(df2$perr)

############################### Keras ###################

library(keras)
library(tensorflow)

# Create preprocessing steps
df_num = df[, c('carat', 'depth', 'table', 'volume')]
df_cat = df[, c('cut', 'color', 'clarity')]

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
maxmindf <- as.data.frame(lapply(df_num, normalize))
price <- df$price
maxmindf <- cbind(maxmindf, price)

library(fastDummies)
dummydf <- dummy_cols(df_cat)
dummydf$NA_ <- NULL

processdf <- cbind(maxmindf, dummydf)
names(processdf)

processdf$cut <- NULL
processdf$color <- NULL
processdf$clarity <- NULL

names(processdf)

require(dplyr)
processdf = rename(processdf, "cut_Very_Good" = "cut_Very Good")


# Split the data
my.data <- processdf
set.seed(123)
my.data$u <- runif(n=dim(my.data)[1],min=0,max=1)

# Create train/test split;
train.df <- subset(my.data, u<0.80);
test.df  <- subset(my.data, u>=0.80);

train.df$u <- NULL
test.df$u <- NULL

names(train.df)
names(test.df)



# Separate features and target

X_train <- train.df[,c("carat", "depth", "table", "volume", "cut_Fair", "cut_Good",
                       "cut_Very_Good", "cut_Premium", "cut_Ideal", "color_D", "color_E",
                       "color_F", "color_G", "color_H", "color_I", "color_J", "clarity_I1",
                       "clarity_SI2", "clarity_SI1", "clarity_VS2", "clarity_VS1",
                       "clarity_VVS2", "clarity_VVS1", "clarity_IF")]
y_train = train.df["price"]

X_test <- test.df[,c("carat", "depth", "table", "volume", "cut_Fair", "cut_Good",
                       "cut_Very_Good", "cut_Premium", "cut_Ideal", "color_D", "color_E",
                       "color_F", "color_G", "color_H", "color_I", "color_J", "clarity_I1",
                       "clarity_SI2", "clarity_SI1", "clarity_VS2", "clarity_VS1",
                       "clarity_VVS2", "clarity_VVS1", "clarity_IF")]
y_test = test.df['price']

# https://rpubs.com/jmsallan/nn_intro

# Build the neural network model


############################################################
#####################  Neural Nets ######################
#########################################################

bh_model <- keras_model_sequential() %>%
layer_dense(units = 64, activation = "relu",
            input_shape = ncol(X_train)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)

bh_model %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mean_absolute_error"))

fit_bh <- bh_model %>% fit(
  X_train,
  y_train,
  epochs = 500,
  validation_split = 0.1,
  verbose = 0
)

bh_model %>% evaluate(X_train, y_train)

bh_model %>% evaluate(X_test, y_test)

################################################

## https://www.r-bloggers.com/2021/04/deep-neural-network-in-r/ (regression)
## https://www.geeksforgeeks.org/building-a-simple-neural-network-in-r-programming/ (class)
## https://www.learnbymarketing.com/tutorials/neural-networks-in-r-tutorial/
## https://datascienceplus.com/neuralnet-train-and-test-neural-networks-using-r/
## https://app.datacamp.com/learn/tutorials/neural-network-models-r


library(neuralnet)

set.seed(333)
nn <- neuralnet(price ~ carat + depth + table + volume + cut_Fair + cut_Good +
                cut_Very_Good + cut_Premium + cut_Ideal + color_D + color_E +
                color_F + color_G + color_H + color_I + color_J + clarity_I1 +
                clarity_SI2 + clarity_SI1 + clarity_VS2 + clarity_VS1 +
                clarity_VVS2 + clarity_VVS1 + clarity_IF,
                data=processdf, 
                hidden= c(8), 
                linear.output=TRUE, 
                threshold=70000.0,
                lifesign = 'full', 
                rep = 1,
                stepmax = 80000)

# plot our neural network  
plot(nn, rep = 1) 

list <- unlist(nn$net.result)

df$pred <- list
df$error <- df$y - df$pred

MAPE <- round(mean(abs((df$y - df$pred)/df$y))*100)
MAPE

# error 
nn$result.matrix 

########################## stop here ###########################




##############################################

ggplot(df_2, aes(x=z, y=rfCoarse)) + 
  geom_point(size=3) +
  ylim(-1,1) +
  xlim(-1,1) +
  geom_abline(intercept=0, slope=1, color = "red", size=2) +
  ggtitle("Actual vs predicted z - RF Coarse - MAPE 91%") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df_2, aes(x=z, y=rfFine)) + 
  geom_point(size=3) +
  ylim(-1,1) +
  xlim(-1,1) +
  geom_abline(intercept=0, slope=1, color = "red", size=2) +
  ggtitle("Actual vs predicted z - RF Fine - MAPE 53%") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df_2, aes(x=z, y=rfSim)) + 
  geom_point(size=3) +
  ylim(-1,1) +
  xlim(-1,1) +
  geom_abline(intercept=0, slope=1, color = "red", size=2) +
  ggtitle("Actual vs predicted z - RF Bootstrap - MAPE 87%") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df_2, aes(x=z, y=knn)) + 
  geom_point(size=3) +
  ylim(-1,1) +
  xlim(-1,1) +
  geom_abline(intercept=0, slope=1, color = "red", size=2) +
  ggtitle("Actual vs predicted z - KNN - MAPE 59%") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(df_2, aes(x=z, y=neuralnet)) + 
  geom_point(size=3) +
  ylim(-1,1) +
  xlim(-1,1) +
  geom_abline(intercept=0, slope=1, color = "red", size=2) +
  ggtitle("Actual vs predicted z - Neural Net - MAPE 10%") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#################### t sne plot - fine ##################

library(Rtsne)
#iris_unique <- unique(iris) # Remove duplicates
df <- unique(df)
dfsub <- df[, c("carat","table","depth", "volume", "price")]
dfsub<- unique(dfsub)
Finedf_matrix <- as.matrix(dfsub)

set.seed(42) # Set a seed if you want reproducible results
tsne_out <- Rtsne(Finedf_matrix) # Run TSNE

# Show the objects in the 2D tsne representation
plot(tsne_out$Y)

library(ggplot2)
tsne_plot <- data.frame(xtsne = tsne_out$Y[,1], ytsne = tsne_out$Y[,2])
ggplot(tsne_plot) + geom_point(aes(x=xtsne, y=ytsne)) +
  ggtitle("tSNE plot") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))


#####################################

library("scatterplot3d")
scatterplot3d(df_2$x, df_2$y, df_2$z, pch=16, color = "blue", zlim=c(-1, 1),
              main = "3D plot of Actual data - Fine grid")
scatterplot3d(df_2$x, df_2$y, df_2$z, pch=16, type="h", color="blue", zlim=c(-1, 1),
              main = "3D plot of Actual data - Fine grid")

scatterplot3d(df_2$x, df_2$y, df_2$neuralnet, pch=16, color = "red", zlim=c(-1, 1),
              main = "3D plot of Neural Net Predicted data - Fine grid")
scatterplot3d(df_2$x, df_2$y, df_2$neuralnet, pch=16, type="h", color = "red",zlim=c(-1, 1),
              main = "3D plot of Neural Net Predicted data - Fine grid")

