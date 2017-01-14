import pandas as pd
import numpy as np


docs = pd.read_csv("PredictRatings.csv", header=None)
observed = docs.values[:]
docs = pd.read_csv("netflix/TestingRatings.csv", header=None)
real = docs.values[:]

tmp = 0
for i,j in observed, real:
    tmp = np.abs(j[2]-i[2])
print(tmp = tmp / len(real))


txt_file = open("RecommendMovie.txt", "w")
for i in observed:
    if i[2] >= 4:
        txt_file.write("{0},{1},{2}", int(i[0]), int(i[1]), i[2])
txt_file.close()

