##############################################################################
## libraries
.libPaths()
.libPaths("C:/Program Files/R/R-4.5.1/library")
.libPaths()

#install.packages(c("rpart", "rpart.plot", "caTools","ggplot2"))
library(rpart)  # For decision tree model
library(rpart.plot)  # For data visualization
library(caTools)  # For data splitting
library(ggplot2)  # For data visualization

##############################################################################
#Regression Tree
## Dataset
Baseball <- read.table("D:/學校資料/輔仁大學/大四上/機器學習/ball.txt", header = TRUE, sep = "", quote = "\"", dec = ".")
head(Baseball)
dim(Baseball)

split <- sample.split(Baseball, SplitRatio = 0.8)
train_data <- subset(Baseball, split == "TRUE")
test_data <- subset(Baseball, split == "FALSE")
dim(train_data)
dim(test_data)
## Regression Tree (Without Categorical variable)
fit1 <- rpart(Logsalary ~ Bat86 + Hit86 + Hr86 + Run86 + 
               Rb86 + Wlk86 + Yrs + Batcr + 
               Hitcr + Hrcr + Runcr + Rbcr +
               Wlkcr +  League86 +  Div86 + 
               Puto86 +  Asst86 + Err86 +
               League87,
               method = "anova",
               data = Baseball)
rpart.plot(fit1)

fit1$variable.importance #變數之重要性

pred.fit1.tree = predict(fit1, test_data)
mse <- mean((pred.fit1.tree - test_data$Logsalary)^2)
mse
data.frame(pred.Logsalary = pred.fit1.tree, true.Logsalary = test_data$Logsalary)
qqplot(pred.fit1.tree, test_data$Logsalary,
     xlim = c(4, 8),
     ylim = c(4, 8),
     xlab = "Predicted Values",
     ylab = "Observed Values",
     main = "Regression Tree",
     pch = 19)

abline(a = 0, b = 1, col = "red", lty = 2)

## Regression Tree (With Categorical variable)
fit2 <- rpart(Logsalary ~ Bat86 + Hit86 + Hr86 + Run86 + 
                         Rb86 + Wlk86 + Yrs + Batcr + 
                         Hitcr + Hrcr + Runcr + Rbcr +
                         Wlkcr +  League86 +  Div86 + Team86 +
                         Pos86 +  Puto86 +  Asst86 + Err86 +
                         League87 +  Team87,
                         data = Baseball)
rpart.plot(fit2)
