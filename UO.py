
import urllib.request
import requests
from bs4 import BeautifulSoup
import os
import numpy as np

""" INPUT PARAMETERS """
store = 'UO'
folder_name = 'scraped'
user = 'annie'

"""*****************"""

pages = 10

if user != 'yanisa':
    search_terms = [["red"], ['black'], ['white'], ['yellow'], ['orange'], ['blue'], ['green'], ['buttons'],['no buttons']]

colors = ["red",'black','white','yellow','orange','blue','green']
color_url_base = 'https://www.urbanoutfitters.com/womens-tops?color='

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
        url +=[temp + search_terms[i][j+1] + '&d=Womens']
        print('# ', i, ': ', url[len(url) - 1])
        for k in range(2,pages+1):
            url += [temp + '&d=Womens&pageNum=' + str(k)]
            print('# ', i, ': ', url[len(url) - 1])
else:
    for i in range(0,len(colors)):
        for j in range(1,pages+1):
            if j == 1:
                url += [color_url_base + colors[i]]
            else:
                url += [color_url_base + colors[i] + '&page=' + str(j) ]
            print('# ',i, ': ', url[len(url)-1])
    for j in range(1,pages+1):
        if j == 1:
            url += ['https://www.urbanoutfitters.com/search?page=2&q=buttons+tops']
        else:
            url += ['https://www.urbanoutfitters.com/search?page=' + str(j) + '&q=buttons+tops']
        # print('# ', i+1, ': ', url[len(url) - 1])
    for j in range(1,pages+1):
        url += ['https://www.urbanoutfitters.com/womens-tops?page=' + str(j)]
        # print('# ', i+2, ': ', url[len(url) - 1])


tally = []
num_in_page = [0]*(pages+1)
bigcnt = 0
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
    imgs = soup.findAll('div', {
        'class': "dom-product-tile c-product-tile c-product-tile--regular c-product-tile js-product-tile"})
    for img in imgs:
        try:
            img_urls += ['https:' + img.find('img')['data-src']]
            print('Image Found: ', 'https:' + img.find('img')['data-src'])
        except:
            pass

    path = './'+folder_name +'/' + store + '/' + label
    print('Path: ',path)

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    # print(num_in_page)
    temp = i//pages+1
    for n, data in enumerate(img_urls):

        print('# ', bigcnt, data)
        try:
            urllib.request.urlretrieve(data,f"{path}/{bigcnt}.jpg")
            bigcnt += 1
        except:
            pass