

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


label_encoder.fit(['US', 'CN', 'ID'])
d = label_encoder.transform(['US', 'CN',  'CN'])
print(d)

