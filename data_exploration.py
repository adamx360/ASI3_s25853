
import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych
data = pd.read_csv('CollegeDistance.csv')

# Eksploracja danych
print(data.info())
print(data.describe())

# Histogram zmiennej celu 'score'
plt.figure(figsize=(8, 6))
plt.hist(data['score'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of 'score' Variable")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.savefig('score_histogram.png')
plt.close()
