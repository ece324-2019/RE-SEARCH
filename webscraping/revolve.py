
import urllib.request
import requests
from bs4 import BeautifulSoup
import os
import numpy as np

""" INPUT PARAMETERS """
store = 'revolve'
folder_name = 'data'
user = 'yanisa'

num_desired = 300
search_terms = [['turtle','neck', 'top'], ['v', 'neck', 'top'], ['collared', 'top'], ['crew', 'neck', 'top'],
                ['square', 'neck', 'top'], ['scoop', 'neck', 'top']]
"""*****************"""
search_terms = [['long','sleeve','top'],['short','sleeve','top'],['sleeveless','top']]


pages = num_desired//87 + 1

if user != 'yanisa':
    search_terms = [["red"], ['black'], ['white'], ['yellow'], ['orange'], ['blue'], ['green'], ['buttons'],['no buttons']]

colors = ["red",'black','white','yellow','orange','blue','green']
color_url_base = 'https://www.revolve.com/tops/br/db773d/?searchsynonym='
in_between = '+top&pageNum='
add = '&color%5B%5D='

if os.path.exists('./'+folder_name +'/'+store):
    pass
else:
    os.mkdir('./'+folder_name +'/'+store)

print('Store: ',store)
url = []

if user == 'yanisa':
    #https://www.revolve.com/r/Search.jsp?search=sleeveless+top&d=Womens
    for i in range(0,len(search_terms)):
        temp = 'https://www.revolve.com/r/Search.jsp?search='
        for j in range(0,len(search_terms[i])-1):
            temp = temp + search_terms[i][j] + '+'
        url +=[temp + '&d=Womens']
        print('# ', i, ': ', url[len(url) - 1])
        for k in range(2,pages+1):
            url += [temp + '&d=Womens&pageNum=' + str(k)]
            print('# ', i, ': ', url[len(url) - 1])
else:
    for i in range(0,len(colors)):
        for j in range(1,pages+1):
            if j == 1:
                url += ['https://www.revolve.com/tops/br/db773d/?searchsynonym=' + colors[i] + '+top&color%5B%5D=' + colors[i]]
            else:
                url += [color_url_base + colors[i] + in_between + str(j) + add + colors[i]]
            print('# ',i, ': ', url[len(url)-1])
    for j in range(1,pages+1):
        if j == 1:
            url += ['https://www.revolve.com/r/Search.jsp?search=tops+with+buttons&d=Womens']
        else:
            url += ['https://www.revolve.com/r/Search.jsp?search=tops+with+buttons&d=Womens&pageNum=' + str(j)]
        print('# ', i+1, ': ', url[len(url) - 1])
    for j in range(1,pages+1):
        url += ['https://www.revolve.com/tops/br/db773d/?searchsynonym=tops&pageNum=' + str(j)]
        print('# ', i+2, ': ', url[len(url) - 1])


tally = []
num_in_page = [0]*(pages+1)
# print(num_in_page)
for i in range(0,len(url)):

    label = search_terms[i//pages]
    label = ' '.join(label)

    print('\nLooking for (',label,')')
    result = requests.get(url[i], headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
    print('Status code:',result.status_code)
    if result.status_code == 200:
        soup = BeautifulSoup(result.content, "html.parser")

    img_urls = []
    imgs = soup.findAll('img', {'class': 'products-grid__image-link-img plp-image js-plp-image js-plp-lazy-img'})
    # print('how many did you find?',len(imgs))
    for img in imgs:
        # print(img)
        try:
            img_urls += [img['data-src']]
            num_in_page[i+1] += 1
        except:
            pass

    path = './'+folder_name +'/' + store + '/' + label
    print('Path: ',path)

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    temp = i//pages+1
    for n, data in enumerate(img_urls):

        print('# ', (n+sum(num_in_page[0:i+1]))/temp, data)
        try:
            urllib.request.urlretrieve(data,f"{path}/{(n+sum(num_in_page[0:i+1]))/temp}.jpg")
        except:
            pass