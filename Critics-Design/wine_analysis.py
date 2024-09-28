import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import StringIO

def read_data(filename):
    with open(filename, "r") as f:
        data = f.read()
    return data

filename = "Datas/wine.txt"

wine_data = pd.read_csv(StringIO(read_data(filename)), header=None)
wine_data.columns = ['Variety', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 
                     'Magnesium', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 
                     'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315', 'Proline']

corr_matrix = wine_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Matrice de Corrélation')
plt.tick_params(rotation=25)
plt.savefig("Figures/correlation_matrix.pdf")
# plt.show()


nb_feats = 4
best_feats = abs(corr_matrix['Variety']).sort_values(ascending=False)[1:nb_feats + 1].index.array

pair_plot = sns.pairplot(wine_data, hue='Variety', vars=best_feats, palette='Set2')
pair_plot.fig.suptitle("Scatterplot Matrix of Selected Wine Features", y=1.02)
plt.savefig("Figures/scatterplot_matrix.pdf")
# plt.show()