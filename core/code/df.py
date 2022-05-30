import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot


larin = 'gfgfgd'
data = pd.read_csv('core\data\data.csv', encoding="ISO-8859-1",
                   dtype={'CustomerID': str, 'InvoiceID': str})

# modifing invoiceDate type to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
print("data.csv is loaded")

# _____________________________________________________________________________
# data cleaning:
# ___________________
# removing null values:
# ___________________
# gives some infos on columns types and numer of null values
tab_info = pd.DataFrame(data.dtypes).T.rename(index={0: 'column type'})
tab_info = tab_info.append(pd.DataFrame(
    data.isnull().sum()).T.rename(index={0: 'null values (nb)'}))
tab_info = tab_info.append(pd.DataFrame(data.isnull().sum()/data.shape[0]*100).T.
                           rename(index={0: 'null values (%)'}))

data.dropna(axis=0, subset=['CustomerID'], inplace=True)
print('Dataframe dimensions:', data.shape)

# ___________________
# removing Duplicate entries:
# ___________________
print('Duplicate entries: {}'.format(data.duplicated().sum()))
data.drop_duplicates(inplace=True)

# ___________________
# data table:
# ___________________

tbl = go.Figure(data=[go.Table(
    #columnwidth = 80,
    header=dict(values=list(data.columns),
                fill_color='bisque',
                ),
    cells=dict(values=[data[data.columns[i]][:15] for i in range(len(data.columns))],
               fill_color='white',
               ))
])
tbl.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    width=850,
    height=350,
)

# tbl.show(renderer='colab')
tbl = tbl.to_json()

# _____________________________________________________________________________
# countries overview:

# ___________________
# no. of countries:
# ___________________
temp = data[['CustomerID', 'InvoiceNo', 'Country']].groupby(
    ['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop=False)
countries = temp['Country'].value_counts()
print('Number. of countries in the dataframe: {}'.format(len(countries)))

# ___________________
# countries heatmap:
# ___________________
hdata = dict(type='choropleth',
             locations=countries.index,
             locationmode='country names', z=countries,
             text=countries.index, colorbar={'title': 'Order nb.'},
             colorscale=[[0, 'rgb(243,198,241)'],
                         [0.01, 'rgb(191,116,250)'], [0.02, 'rgb(31,120,180)'],
                         [0.03, 'rgb(255,202,113)'], [0.05, 'rgb(51,160,44)'],
                         [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
                         [1, 'rgb(255,92,110)']],
             reversescale=False)
# _______________________
layout = dict(title='',
              geo=dict(showframe=False, projection={'type': 'mercator'}))
# ______________
choromap = go.Figure(data=[hdata], layout=layout)
choromap.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    width=400,
    height=320,
)
#iplot(choromap, validate=False)
# choromap.show(renderer="colab")
heatmap = choromap.to_json()

# ___________________
# total no of Customers, products and transaction:
# ___________________
total = {'products': len(data['StockCode'].value_counts()),
         'transactions': len(data['InvoiceNo'].value_counts()),
         'customers': len(data['CustomerID'].value_counts()),
         }

# ___________________
# no of products in each transaction:
# ___________________
temp = data.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[
    'InvoiceDate'].count()
nb_products_per_basket = temp.rename(
    columns={'InvoiceDate': 'Number of products'})

# ___________________
# cancelling orders:
# ___________________

nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(
    lambda x: int('C' in x))

# ------------------------------------------------------------------------------______________________________________________

# ___________________
# importing data after cleanig it by removing null values , cancelled orders and doubtfull entries :
# ___________________

df_cleaned = pd.read_csv('core\data\df_cleaned.csv')
# dropping the first column
df_cleaned = df_cleaned.iloc[:, 1:]


# ___________________
# stock code:
# ___________________
list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains(
    '^[a-zA-Z]+', regex=True)]['StockCode'].unique()

# ___________________
# the total price of every purchase:
# ___________________

df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * \
    (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])

# ___________________________________________
# sum of purchases / user & order
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[
    'TotalPrice'].sum()
basket_price = temp.rename(columns={'TotalPrice': 'Basket Price'})

# _____________________
# date of the order

df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[
    'InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis=1, inplace=True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
# ______________________________________
# selection of significant inputs:
basket_price = basket_price[basket_price['Basket Price'] > 0]

# ____________________
# piechart : Representation of the number of purchases / amount
# ____________________

# Count of purchases
price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
count_price = []
for i, price in enumerate(price_range):
    if i == 0:
        continue
    val = basket_price[(basket_price['Basket Price'] < price) &
                       (basket_price['Basket Price'] > price_range[i-1])]['Basket Price'].count()
    count_price.append(val)
# ____________________________________________
# Representation of the number of purchases / amount
sizes = count_price
labels = ['0<.<50', '50<.<100', '100<.<200', '200<.<500',
          '500<.<1000', '1000<.<5000', '5000<.<50000']
pie_df = pd.DataFrame(sizes, index=labels)
# print(count_price)
colors = ['#fe919d', '#ffca71', '#c077f9',
          '#ffe1bc', '#f4c7ef', 'purple', 'lightblue', ]

pie_chart = go.Figure(data=[go.Pie(
    labels=labels, values=sizes, hole=.3)])

pie_chart.update_traces(hoverinfo='label+percent', textinfo='value',
                        marker=dict(colors=colors))

pie_chart.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    width=400,
    height=285,
)
piechart = pie_chart.to_json()
# pie_chart.show(renderer="colab")
