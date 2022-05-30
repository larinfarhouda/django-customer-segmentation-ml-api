from datetime import datetime
from datetime import date
import pandas as pd
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from plotly.graph_objs import *

from core.code import df
from core.code import productscat

# ____________________
#  Formatting data
# ____________________

# create the categorical variable categ_product where I indicate the cluster of each product :
corresp = dict()
for key, val in zip(productscat.liste_produits, productscat.clusters):
    corresp[key] = val
# __________________________________________________________________________
df.df_cleaned['categ_product'] = df.df_cleaned.loc[:,
                                                   'Description'].map(corresp)
larin = 'uuuuuuuuuu'


# ____________________
#  Grouping products
# ____________________

# create the categ_N variables (with  N∈[0:4] ) that contains the amount spent in each product category:
for i in range(5):
    col = 'categ_{}'.format(i)
    df_temp = df.df_cleaned[df.df_cleaned['categ_product'] == i]
    price_temp = df_temp['UnitPrice'] * \
        (df_temp['Quantity'] - df_temp['QuantityCanceled'])
    price_temp = price_temp.apply(lambda x: x if x > 0 else 0)
    df.df_cleaned.loc[:, col] = price_temp
    df.df_cleaned[col].fillna(0, inplace=True)
# __________________________________________________________________________________________________
#df_cleaned[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3','categ_4']][:5]
pd.to_numeric(df.df_cleaned['TotalPrice'])

# ___________________________________________
# sum of purchases / user & order
temp = df.df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[
    'TotalPrice'].sum()
basket_price = temp.rename(columns={'TotalPrice': 'Basket Price'})

# ____________________________________________________________
# percentage of order price / product category
for i in range(5):
    col = 'categ_{}'.format(i)
    temp = df.df_cleaned.groupby(
        by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
    basket_price.loc[:, col] = temp[col]
# _____________________
# date of the order
df.df_cleaned['InvoiceDate_int'] = df.df_cleaned['InvoiceDate'].astype('int64')
temp = df.df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[
    'InvoiceDate_int'].mean()
df.df_cleaned.drop('InvoiceDate_int', axis=1, inplace=True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
# ______________________________________
# selection of significant inputs:
basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID', ascending=True)[:5]


# ====================================================================================

# ____________________
#  Separation of data over time
# ____________________

# plit the data set by retaining the first 10 months to develop the model and the following two months to test it:
date = date(2011, 10, 1)
my_time = datetime.min.time()
set_entrainement = basket_price[basket_price['InvoiceDate'] < datetime.combine(
    date, my_time)]
set_test = basket_price[basket_price['InvoiceDate']
                        >= datetime.combine(date, my_time)]
basket_price = set_entrainement.copy(deep=True)

# ___________________________________________________
#  Consumer Order Combinations
# ___________________________________________________

# grouping together the different entries that correspond to the same user
# ________________________________________________________________
# number of visits and stats on the amount of the basket / users
transactions_per_user = basket_price.groupby(
    by=['CustomerID'])['Basket Price'].agg(['count', 'min', 'max', 'mean', 'sum'])
for i in range(5):
    col = 'categ_{}'.format(i)
    transactions_per_user.loc[:, col] = basket_price.groupby(by=['CustomerID'])[col].sum() /\
        transactions_per_user['sum']*100

transactions_per_user.reset_index(drop=False, inplace=True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending=True)[:5]


# two additional variables that give the number of days elapsed since the first purchase (** FirstPurchase ) and the number of days since the last purchase ( LastPurchase **):

last_date = basket_price['InvoiceDate'].max().date()

first_registration = pd.DataFrame(
    basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
last_purchase = pd.DataFrame(basket_price.groupby(
    by=['CustomerID'])['InvoiceDate'].max())

test = first_registration.applymap(lambda x: (last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x: (last_date - x.date()).days)

transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop=False)[
    'InvoiceDate']
transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop=False)[
    'InvoiceDate']

# ___________________________________________________
#  Creating the table
# ___________________________________________________

tbll = go.Figure(data=[go.Table(
    #columnwidth = 80,
    header=dict(values=list(transactions_per_user.columns),
                fill_color='bisque',
                ),
    cells=dict(values=[transactions_per_user[transactions_per_user.columns[i]][:25].map("{:.2f}".format) if transactions_per_user[transactions_per_user.columns[i]].dtypes == 'float64' else transactions_per_user[transactions_per_user.columns[i]][:25] for i in range(len(transactions_per_user.columns))],
               fill_color='white', height=30
               ))
])


tbll.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    width=1200,
    height=350,
)

tbll = tbll.to_json()

# ======================================================================================
# Creation of customers categories
# ======================================================================================

# ____________________
#  Data encoding
# ____________________

list_cols = ['count', 'min', 'max', 'mean', 'categ_0',
             'categ_1', 'categ_2', 'categ_3', 'categ_4']
# _____________________________________________________________
selected_customers = transactions_per_user.copy(deep=True)
matrix = selected_customers[list_cols].values

# standardizing the mean of the data
scaler = StandardScaler()
scaler.fit(matrix)
print('variables mean values: \n' + 90*'-' + '\n', scaler.mean_)
scaled_matrix = scaler.transform(matrix)


# create a representation of the different clusters and thus verify the quality of the separation of the different groups by perform a PCA
pca = PCA()
pca.fit(scaled_matrix)
pca_samples = pca.transform(scaled_matrix)


# ____________________
#  Creating customer clusters
# ____________________

n_clusters = 11
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100)
kmeans.fit(scaled_matrix)
clusters_clients = kmeans.predict(scaled_matrix)
silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
#print('score de silhouette: {:<.3f}'.format(silhouette_avg))

# ____________________
#  representation of customer clusters
# ____________________
pd.Series(clusters_clients).value_counts()
cus_clusters = pd.Series(clusters_clients).value_counts()
data = {'no. customers': cus_clusters.values, 'cluster': cus_clusters.index}

barchdata = dict(type="bar", x=data['cluster'],
                    y=data['no. customers'])

figbar = go.Figure(data=[barchdata])
figbar.update_layout(
    title_text="no. of customers in each cluster",  xaxis_title="cluster",
    yaxis_title="no. of customers")

figbar.update_traces(marker_color='#ff919d')

figbar.update_layout(
    title_text='representation of the most common keywords in products description',
    margin=dict(l=20, r=10, t=40, b=10),
    width=650,
    height=245,
    plot_bgcolor="#fff1e0",
)

#figbar.show(renderer = 'colab')
figbar = figbar.to_json()
# ____________________
# understand the habits of the customers in each cluster
# ____________________

# adding to the 'selected_customers' dataframe a variable that defines the cluster to which each client belongs:
selected_customers.loc[:, 'cluster'] = clusters_clients

#

merged_df = pd.DataFrame()
for i in range(n_clusters):
    test = pd.DataFrame(
        selected_customers[selected_customers['cluster'] == i].mean())
    test = test.T.set_index('cluster', drop=True)
    test['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
    merged_df = pd.concat([merged_df, test])
# _____________________________________________________
merged_df.drop('CustomerID', axis=1, inplace=True)
print('number of customers:', merged_df['size'].sum())

merged_df = merged_df.sort_values('sum')


# re-organize the content of the dataframe by ordering the different clusters: first, in relation to the amount wpsent in each product category and then, according to the total amount spent:

liste_index = []
for i in range(5):
    column = 'categ_{}'.format(i)
    liste_index.append(merged_df[merged_df[column] > 45].index.values[0])
# ___________________________________
liste_index_reordered = liste_index
liste_index_reordered += [s for s in merged_df.index if s not in liste_index]
# ___________________________________________________________
merged_df = merged_df.reindex(index=liste_index_reordered)
merged_df = merged_df.reset_index(drop=False)
