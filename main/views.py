#from django.shortcuts import render
import code
from rest_framework.response import Response
from rest_framework.decorators import api_view
from core.code import df
from core.code import productscat
from core.code import customerscat
from core.code import predction

# Create your views here.


@api_view(['GET'])
def dataprep(request):
    datainfo = {'data': df.tbl,
                'dimentions': df.data.shape,
                'heatmap': df.heatmap,
                'totals': df.total,  # total no of Customers, products and transaction
                # no of products in each transaction and the canceld orders
                'productsInTransaction': df.nb_products_per_basket[:10].sort_values('CustomerID'),
                # canceld orders percentage
                'canceldOrders': ["3654/22190", "16.47%"],
                # selection of significant inputs
                'significantInputs': df.basket_price.sort_values('CustomerID')[:10],
                'piechart': df.piechart}  # piechart : Representation of the number of purchases / amount
    info = {"info": datainfo}
    return Response(info)


@api_view(['GET'])
def prodcat(request):
    datainfo = {'Wordschart': productscat.barchart,  # representation of the most common keywords in products description
                'Range': productscat.piechartt,
                'Barchart': productscat.figbar,
                }
    info = {"info": datainfo}
    return Response(info)


@api_view(['GET'])
def cuscat(request):
    datainfo = {'customers': customerscat.tbll,  # how customers spend their money on different categories
                # nb. customers with one-time purchase
                'purchase': ['1445/3608', '40.05%'],
                'cluster': customerscat.figbar}  # nb. customers in each cluster
    info = {"info": datainfo}
    return Response(info)


@api_view(['GET'])
def pred(request):
    datainfo = {'presentation of Classifiers Precision percentage': predction.figbarr,
                }
    return Response(datainfo)
