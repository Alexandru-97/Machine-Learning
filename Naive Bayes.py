from PIL import Image
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import csv
import pandas as pnd
from sklearn.metrics import confusion_matrix


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

# Antrenez pentru imaginile de train cu modelul Bayes Naiv folosind parametrii default
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_images, train_labels)

# obtin acuratetea modelului pentru imaginile de validare
score = naive_bayes_model.score(validation_images, validation_labels)

# verific clasificarea obtinuta de model pentru datele de validare
p = naive_bayes_model.predict(validation_images)
print(score)

ids = []

print(confusion_matrix(validation_labels, p))

with open('test.txt') as f:
    for line in f.readlines():
        ids.append(line.rstrip("\n"))

output = pnd.DataFrame({'id': ids,
                    'label': p})
output.to_csv('output.csv', index=False)