##################### Load Packages (For Analysis) ####################
## Libraries Path
.libPaths()
.libPaths("C:/Program Files/R/R-4.5.1/library")
.libPaths()

## Install Packages (Run Once)
#install.packages(c("randomForest", "caTools", "ggplot2","GGally", "gridExtra", "mlbench"))

## Load Packages
library(randomForest) # For Random Forest
library(caTools)      # For Data Splitting
library(ggplot2)      # For Data Visualization
library(GGally)       # For Scatter Plot Matrix
library(gridExtra)    # For Plot Arrangement
library(mlbench)      # For Boston Housing Data

setwd("D:/學校資料/輔仁大學/大四上/機器學習")

#############################################################################
## Random Forest
## Boston Housing Data (For Regression)
#############################################################################

## Load Data
Baseball <- read.table("D:/學校資料/輔仁大學/大四上/機器學習/ball.txt", header = TRUE, sep = "", quote = "\"", dec = ".")

dim(Baseball)
summary(Baseball)
str(Baseball)
head(Baseball, n = 10)

########################################################################
## Data Preparation
########################################################################

## 只保留反應變數 Logsalary 與 22 個解釋變數
## （其餘欄位如球員姓名、原始薪資等會被刪除，避免放進 Logsalary ~ . ）
vars <- c("Logsalary",
          "Bat86", "Hit86", "Hr86", "Run86", "Rb86", "Wlk86", "Yrs",
          "Batcr", "Hitcr", "Hrcr", "Runcr", "Rbcr", "Wlkcr",
          "League86", "Div86", "Team86", "Pos86",
          "Puto86", "Asst86", "Err86", "League87", "Team87")
Baseball <- Baseball[, vars]

## 刪除缺失值（薪資有 NA 的球員）
Baseball <- na.omit(Baseball)

## 類別變數轉成 factor（randomForest 不接受文字欄位）
cat_vars <- c("League86", "Div86", "Team86", "Pos86", "League87", "Team87")
Baseball[cat_vars] <- lapply(Baseball[cat_vars], factor)

dim(Baseball)
str(Baseball)

## 合併人數少的守備位置（避免測試集出現訓練集沒有的層次）
tab <- table(Baseball$Pos86)
Baseball$Pos86 <- as.character(Baseball$Pos86)
Baseball$Pos86[Baseball$Pos86 %in% names(tab)[tab < 5]] <- "Other"
table(Baseball$Pos86)

########################################################################
## Training Data and Testing Data
########################################################################

set.seed(123)

split <- sample.split(Baseball$Logsalary,SplitRatio = 0.8)

train_data <- subset(Baseball, split == TRUE)
test_data  <- subset(Baseball, split == FALSE)

dim(train_data)
dim(test_data)

## 檢查：測試集是否有訓練集沒出現過的類別（會讓 glm 的 predict 報錯）
sapply(cat_vars, function(v)
  setdiff(unique(as.character(test_data[[v]])),
          unique(as.character(train_data[[v]]))))

########################################################################
## Multiple Linear Regression
########################################################################

lm.fit <- glm(
  Logsalary ~ .,
  family = gaussian(link = "identity"),
  data = train_data
)

summary(lm.fit)

## Prediction
pred.lm.fit <- predict(
  lm.fit,
  newdata = test_data
)

## Mean Squared Error
mse.lm.fit <- mean(
  (pred.lm.fit - test_data$Logsalary)^2
)
mse.lm.fit

## Root Mean Squared Error
rmse.lm.fit <- sqrt(mse.lm.fit)
rmse.lm.fit

## Predicted vs. Observed Values
data.frame(
  pred.Logsalary = pred.lm.fit,
  true.Logsalary = test_data$Logsalary
)

windows()
plot(pred.lm.fit, test_data$Logsalary,
     xlim = c(4, 8),
     ylim = c(4, 8),
     xlab = "Predicted Values",
     ylab = "Observed Values",
     main = "Multiple Linear Regression",
     pch = 19)

abline(a = 0, b = 1, col = "red", lty = 2)


########################################################################
## Effect of mtry on Random Forest（先用 OOB MSE 挑選最佳 mtry）
########################################################################

mtry.values <- 1:22          # 共 22 個解釋變數
oob.mse <- numeric(length(mtry.values))

for (i in seq_along(mtry.values)) {
  
  set.seed(123)
  
  fit <- randomForest(
    Logsalary ~ .,
    data = train_data,
    ntree = 500,
    mtry = mtry.values[i]
  )
  
  oob.mse[i] <- tail(fit$mse, 1)
}

windows()
plot(mtry.values, oob.mse,
     type = "b",
     pch = 19,
     xlab = "Number of Variables (mtry)",
     ylab = "OOB Mean Squared Error",
     main = "Effect of mtry on Random Forest")

## OOB MSE 最小的 mtry
best.mtry <- mtry.values[which.min(oob.mse)]
best.mtry
min(oob.mse)


########################################################################
## Random Forest Regression（用最佳 mtry 建模）
########################################################################

set.seed(123)

rf.fit <- randomForest(
  Logsalary ~ .,
  data = train_data,
  ntree = 500,
  mtry = best.mtry,
  nodesize = 5,
  importance = TRUE
)

## Model Summary
print(rf.fit)
summary(rf.fit)

## OOB Mean Squared Error
rf.fit$mse
tail(rf.fit$mse, 1)    # 500 棵樹後的 OOB MSE
tail(rf.fit$rsq, 1)    # 500 棵樹後的 OOB R-squared

## OOB Error Plot
windows()
plot(rf.fit,
     main = "Random Forest Regression: OOB MSE")

## Variable Importance
importance(rf.fit)

windows()
varImpPlot(rf.fit,
           main = "Variable Importance: Baseball")

## Prediction
pred.rf.fit <- predict(
  rf.fit,
  newdata = test_data
)

## Mean Squared Error
mse.rf.fit <- mean(
  (pred.rf.fit - test_data$Logsalary)^2
)
mse.rf.fit

## Root Mean Squared Error
rmse.rf.fit <- sqrt(mse.rf.fit)
rmse.rf.fit

## Predicted vs. Observed Values
data.frame(
  pred.Logsalary = pred.rf.fit,
  true.Logsalary = test_data$Logsalary
)

windows()
plot(pred.rf.fit, test_data$Logsalary,
     xlim = c(4, 8),
     ylim = c(4, 8),
     xlab = "Predicted Values",
     ylab = "Observed Values",
     main = "Random Forest Regression",
     pch = 19)

abline(a = 0, b = 1, col = "red", lty = 2)

## 預測誤差最大的球員（兩個模型都猜錯的離群值）
test_data[which.max(abs(pred.rf.fit - test_data$Logsalary)), ]


########################################################################
## Model Comparison
## Multiple Linear Regression vs. Random Forest
########################################################################

## MSE and RMSE Comparison
results <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  MSE = c(mse.lm.fit, mse.rf.fit),
  RMSE = c(rmse.lm.fit, rmse.rf.fit)
)

print(results)

## Visualization of Prediction Results
p1 <- ggplot(
  data.frame(
    Predicted = pred.lm.fit,
    Observed = test_data$Logsalary
  ),
  aes(x = Predicted, y = Observed)
) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1,
              color = "red", linetype = 2) +
  coord_fixed(xlim = c(4, 8), ylim = c(4, 8)) +
  labs(title = "Linear Regression") +
  theme_bw()

p2 <- ggplot(
  data.frame(
    Predicted = pred.rf.fit,
    Observed = test_data$Logsalary
  ),
  aes(x = Predicted, y = Observed)
) +
  geom_point(color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1,
              color = "red", linetype = 2) +
  coord_fixed(xlim = c(4, 8), ylim = c(4, 8)) +
  labs(title = "Random Forest") +
  theme_bw()

windows()
grid.arrange(p1, p2, ncol = 2)
########################################################################
## 隨機森林的重要參數
# ntree = 500         建立 500 棵決策樹
# mtry = 2            每次節點分裂隨機選取 2 個候選變數
# nodesize = 5        終端節點的最小樣本數設定
# importance = TRUE   計算變數重要性
# rf.fit$mse          各棵樹累積後的 OOB MSE
# varImpPlot()        繪製變數重要性圖
# predict()           預測測試資料
# mtry 預設值         分類 為 sqrt(p)，迴歸為 max{1,floor(p/3)}。
########################################################################
