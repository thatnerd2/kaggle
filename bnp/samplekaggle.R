train = read.csv('~/Projects/kaggle/bnp/train.csv')
#test  = read.csv('~/Projects/kaggle/bnp/test.csv')

for (i in 1:ncol(train)) {
  if (is.numeric(train[1, i])) {
    train[, i][is.na(train[, i])] = median(train[, i], na.rm = TRUE) 
  }
}

reg = glm(target ~ ., data = train[, 2:ncol(train)], family = "binomial")
