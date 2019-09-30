import pandas as pd


Folder_Path = r'/Users/yqh/Desktop/'
SaveFile_Path = r'/Users/yqh/Desktop'
SaveFile_Name1 = r'norm_training_data.csv'
SaveFile_Name2 = r'norm_testing_data.csv'

a = pd.read_csv(Folder_Path + 'testing_data.csv', usecols=range(0, 22), header=0)
a1 = pd.read_csv(Folder_Path + 'testing_data.csv', usecols=[22], header=0)
a2 = pd.read_csv(Folder_Path + 'testing_data.csv', usecols=[23], header=0)
a3 = pd.read_csv(Folder_Path + 'testing_data.csv', usecols=[24], header=0)
a4 = pd.read_csv(Folder_Path + 'testing_data.csv', usecols=[25], header=0)
a = (a - a.min()) / (a.max() - a.min())
a.insert(22, 'Family', a1)
a.insert(23, 'Genus', a2)
a.insert(24, 'Species', a3)
a.insert(25, 'RecordID', a4)
a.to_csv(Folder_Path + SaveFile_Name2, index=False, header=a.columns)

b = pd.read_csv(Folder_Path + 'training_data.csv', usecols=range(0, 22), header=0)
b1 = pd.read_csv(Folder_Path + 'training_data.csv', usecols=[22], header=0)
b2 = pd.read_csv(Folder_Path + 'training_data.csv', usecols=[23], header=0)
b3 = pd.read_csv(Folder_Path + 'training_data.csv', usecols=[24], header=0)
b4 = pd.read_csv(Folder_Path + 'training_data.csv', usecols=[25], header=0)
b = (b - b.min()) / (b.max() - b.min())
b.insert(22, 'Family', b1)
b.insert(23, 'Genus', b2)
b.insert(24, 'Species', b3)
b.insert(25, 'RecordID', b4)
b.to_csv(Folder_Path + SaveFile_Name1, index=False, header=a.columns)