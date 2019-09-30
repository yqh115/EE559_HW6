import pandas as pd
from sklearn.model_selection import train_test_split

Folder_Path = r'/Users/yqh/Desktop/'
SaveFile_Path = r'/Users/yqh/Desktop'
SaveFile_Name1 = r'training_data.csv'
SaveFile_Name2 = r'testing_data.csv'

data = pd.read_csv(Folder_Path + 'Frogs_MFCCs.csv', header=0)
X_train, X_test = train_test_split(data, test_size=0.3)

X_train.to_csv(Folder_Path + SaveFile_Name1, index=False, header=data.columns)
X_test.to_csv(Folder_Path + SaveFile_Name2, index=False, header=data.columns)

print(X_train.info)
print(X_test.info)
print(data.columns)
