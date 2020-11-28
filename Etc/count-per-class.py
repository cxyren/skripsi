import pandas as pd

#training data
df = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest4.csv')

print(df['class'].value_counts())