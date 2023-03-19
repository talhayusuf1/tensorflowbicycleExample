from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


# sklearn sayesinde train test ayrimi yaptık ml in bi kutuphanesi
bisikletAnalizDataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")
# Burada xlsx i okuduk

# print(bisikletAnalizDataFrame.head())  # ilk 5
sbn.pairplot(bisikletAnalizDataFrame)  # seaborn ile gosterdik
# plt.show()

# Veriyi test train olarak ikiye ayirmak
# train_test_split()

y = bisikletAnalizDataFrame["Fiyat"].values  # Values Dersek numpy dizisi olur
x = bisikletAnalizDataFrame[["BisikletOzellik1", "BisikletOzellik2"]].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=15)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

scaler = MinMaxScaler()
model = Sequential()

model.add(Dense(4, activation="relu"))
# 3 defa hiddenlayer icin 3 tane ekledim
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))

model.add(Dense(1))  # cıktı 1 tane noron

model.compile(optimizer="rmsprop", loss="mse")
# epochs kac kare calısacagını belirleriz cok fazla gecerse overfitting
model.fit(x_train, y_train, epochs=250)

loss = model.history.history["loss"]
sbn.lineplot(x=range(len(loss)), y=loss)
trainLoss = model.evaluate(x_train, y_train, verbose=0)
testLoss = model.evaluate(x_test, y_test, verbose=0)

print(trainLoss)
print(testLoss)

testTahminleri = model.predict(x_test)
print(testTahminleri)

tahminDataFrame = pd.DataFrame(y_test, columns=["Gercek Y"])
print(tahminDataFrame)


testTahminleri = pd.Series(testTahminleri.reshape(330,))
print(testTahminleri)

tahminDataFrame = pd.concat([tahminDataFrame, testTahminleri], axis=1)
print(tahminDataFrame)

sbn.scatterplot(x="Gercek Y", y="Tahmin Y", data=tahminDataFrame)

print(mean_absolute_error(
    tahminDataFrame["Gercek Y"], tahminDataFrame["Tahmin Y"]))

print(mean_squared_error(
    tahminDataFrame["Gercek Y"], tahminDataFrame["Tahmin Y"]))

bisikletAnalizDataFrame.describe()  # Butun bilgileri bize verir
