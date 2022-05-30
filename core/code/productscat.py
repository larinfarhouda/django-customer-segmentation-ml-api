import pandas as pd
import nltk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import plotly.graph_objs as go
from plotly.graph_objs import *


from core.code import df
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ____________________
# Products Description
# ____________________


def is_noun(pos): return pos[:2] == 'NN'


def keywords_inventory(dataframe, colonne='Description'):
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys = []
    count_keywords = dict()
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s):
            continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(
            tokenized) if is_noun(pos)]

        for t in nouns:
            t = t.lower()
            racine = stemmer.stem(t)
            if racine in keywords_roots:
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1

    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k
                    min_length = len(k)
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]

    print("Nb of keywords in variable '{}': {}".format(
        colonne, len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords


print('done')
# ____________________
# retrieve the list of products:
# ____________________
df_produits = pd.DataFrame(df.data['Description'].unique()).rename(
    columns={0: 'Description'})


# ____________________
# using the function previously defined in order to analyze the description of the various products
# ____________________
keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(
    df_produits)


# ____________________
# representation of the most common keywords
# ____________________

list_products = []
for k, v in count_keywords.items():
    list_products.append([keywords_select[k], v])
list_products.sort(key=lambda x: x[1], reverse=True)

liste = sorted(list_products, key=lambda x: x[1], reverse=True)
# #_______________________________

liste = dict(liste)
words_barchart = go.Figure(
    [go.Bar(x=list(liste.values()), y=list(liste.keys()), orientation="h")])

words_barchart.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                             marker_line_width=1.5, opacity=0.6)

words_barchart.update_layout(
    title_text='representation of the most common keywords in products description',
    margin=dict(l=20, r=10, t=30, b=20),
    width=850,
    height=350,
    plot_bgcolor="#fff1e0",
)
barchart = words_barchart.to_json()


# ____________________
# discard useless words and consider only the words that appear more than 13 times.
# ____________________

list_products = []
for k, v in count_keywords.items():
    word = keywords_select[k]
    if word in ['pink', 'blue', 'tag', 'green', 'orange']:
        continue
    if len(word) < 3 or v < 13:
        continue
    if ('+' in word) or ('/' in word):
        continue
    list_products.append([word, v])
# ______________________________________________________
list_products.sort(key=lambda x: x[1], reverse=True)

# ____________________
# Data encoding
# ____________________

df.df_cleaned['Description'] = df.df_cleaned['Description'].astype('string')
print(df.df_cleaned['Description'].dtypes)
liste_produits = df.df_cleaned['Description'].unique()
# X = pd.DataFrame()
# for key, occurence in list_products:
#     X.loc[:, key] = list(map(lambda x: int(key.upper() in x), liste_produits))

# ____________________
# adding 6 extra columns to X matrix, where I indicate the price range of the products
# ____________________
threshold = [0, 1, 2, 3, 5, 10]
label_col = ['0<.<1', '1<.<2', '2<.<3', '3<.<5', '5<.<10', '.>10']

# for i in range(len(label_col)):
#     X.loc[:, label_col[i]] = 0

# for i, prod in enumerate(liste_produits):
#     prix = df.df_cleaned[df.df_cleaned['Description']
#                          == prod]['UnitPrice'].mean()
#     j = 0
#     while prix > threshold[j]:
#         j += 1
#         if j == len(threshold):
#             break
#     X.loc[i, label_col[j-1]] = 1


# ____________________
#  representation of products in the different groups of range of appearing keywords
# ____________________

prod_range = {'range of appearing keywords': ['0<.<1', '1<.<2', '2<.<3', '3<.<5',
                                              '5<.<10', '.>10'], 'no. products': ['963', '1009', '673', '606', '470', '157']}
colors = ['#fe919d', '#ffca71', '#c077f9',
          '#ffe1bc', '#f4c7ef', 'purple', 'lightblue']

piech = go.Figure(data=[go.Pie(
    labels=prod_range['range of appearing keywords'], values=prod_range['no. products'], hole=.3)])

piech.update_traces(hoverinfo='label+percent', textinfo='value',
                    marker=dict(colors=colors))
piech.update_layout(
    #title_text='presentation of no. of products percentage in the different ranges of appearing keywords',
    margin=dict(l=10, r=10, t=10, b=10),
    width=400,
    height=285,
)

piechartt = piech.to_json()
# piech.show(renderer="colab")
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# ____________________
# Creating clusters of products
# ____________________
X = pd.read_csv('core\data\X.csv')
# dropping the first column
X = X.iloc[:, 1:]
matrix = X.values

n_clusters = 5
silhouette_avg = -1
while silhouette_avg < 0.145:
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)

    #km = kmodes.KModes(n_clusters = n_clusters, init='Huang', n_init=2, verbose=0)
    #clusters = km.fit_predict(matrix)
    #silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# ____________________
# no. of products in each cluster
# ____________________
prodInClus = pd.Series(clusters).value_counts()
prodInClus = {'cluster': [1, 2, 3, 4, 5], 'no. of products': [
    prodInClus[i] for i in range(len(prodInClus))]}

# ____________________
# bar chart representation:
# ____________________

bardata = dict(type="bar", x=prodInClus['cluster'],
                    y=prodInClus['no. of products'])

figbar = go.Figure(data=[bardata])

figbar.update_layout(
    xaxis_title="cluster",
    yaxis_title="no. of products",
    margin=dict(l=10, r=10, t=10, b=10),
    width=400,
    height=285,
    plot_bgcolor="#fff1e0",

)

figbar.update_traces(marker_color='#fe919d')
figbar = figbar.to_json()
