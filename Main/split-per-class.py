import pandas as pd

#training data
df = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest3.csv')

print(df.groupby('class').count().min())