from PIL import Image
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn.datasets as datasets
import numpy as np
import csv
import pandas as pnd


# functie pentru preluarea imaginilor de train, test, validare
def getImages(folder):
    files = []
    images = []

    # retin toate numele fisierelor din folderul respectiv in files
    with open(f'{folder}.txt') as csv_file:
        read_file = csv.reader(csv_file, delimiter=',')
        for row in read_file:
            files.append(row[0])

    # introduc imaginile flattened in array ul images
    for file in files:
        img = Image.open(folder + "/" + file)
        array = np.array(img)
        flatArray = array.ravel()
        images.append(flatArray)

    return images


# functie pentru preluarea etichetelor
def getLabels(folder):
    labels = []
    with open(f'{folder}.txt') as csv_file:
        read_file = csv.reader(csv_file, delimiter=',')
        for row in read_file:
            labels.append(int(row[1]))

    return labels


# incarc datele in variabile
train_labels = getLabels("train")
validation_labels = getLabels("validation")

train_images = getImages("train")
validation_images = getImages("validation")
test_images = getImages("test")

# normalizez imaginile
scaler = preprocessing.StandardScaler()
aux = scaler.fit_transform(train_images)
aux2 = scaler.fit_transform(validation_images)
normalized_train = np.array(aux)
normalized_validation = np.array(aux2)
normalized_test = scaler.transform(test_images)

normalized_concat = np.concatenate((normalized_train, normalized_validation), axis=0)
labels_concat = np.concatenate((train_labels, validation_labels), axis=0)

model = svm.SVC()
model.fit(normalized_concat, labels_concat)

predict = model.predict(normalized_validation)
print(model.score(normalized_validation, validation_labels))
print(confusion_matrix(validation_labels, predict))

ids = []

with open('test.txt') as f:
    for line in f.readlines():
        ids.append(line.rstrip("\n"))

output = pnd.DataFrame({'id': ids,
                        'label': predict})
output.to_csv('output.csv', index=False)