
import urllib.request
import requests
from bs4 import BeautifulSoup
import os
import numpy as np

""" INPUT PARAMETERS """
store = 'nastygal'
folder_name = 'data'
user = 'yanisa'

num_desired = 300
search_terms = [['yellow','top']]
"""*****************"""

pages = 20

if user != 'yanisa':
    #search_terms = [["red"], ['black'], ['white'], ['yellow'], ['orange'], ['blue'], ['green'], ['buttons'],['no buttons']]
    search_terms = [["yellow", "top"]]
        #[["black", "top"], ["white", "top"], ["yellow", "top"], ["green", "top"],["orange", "top"], ["blue", "top"], ["red", "top"]]

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
    for i in range(0,len(search_terms)):
        temp = 'https://www.nastygal.com/ca/search?q='
        for j in range(0,len(search_terms[i])):
            temp = temp + search_terms[i][j] + '+'
            for k in range(0,pages):
                if k == 0:
                    url += [temp ]
                else:
                    url += [temp +  '&sz=80&start=' + str(80*k)]
                # print('# ',i, ': ', url[len(url)-1])


tally = []
num_in_page = [0]*(pages+1)
# print(num_in_page)
bigcnt = 0
# url = ['https://www.nastygal.com/ca/search?q=white+top',
# 'https://www.nastygal.com/ca/search?q=white%20top&sz=80&start=80',
#        'https://www.nastygal.com/ca/search?q=white%20top&sz=80&start=160'
#        'https://www.nastygal.com/ca/search?q=white%20top&sz=80&start=240'
#        'https://www.nastygal.com/ca/search?q=white%20top&sz=80&start=320'
#        'https://www.nastygal.com/ca/search?q=white%20top&sz=80&start=400']

# url = ['https://www.nastygal.com/ca/search?q=yellow+top',
# 'https://www.nastygal.com/ca/search?q=yellow%20top&sz=80&start=80',
#        'https://www.nastygal.com/ca/search?q=yellow%20top&sz=80&start=160'
#        'https://www.nastygal.com/ca/search?q=yellow%20top&sz=80&start=240'
#        'https://www.nastygal.com/ca/search?q=yellow%20top&sz=80&start=320'
#        'https://www.nastygal.com/ca/search?q=yellow%20top&sz=80&start=400']
url = ['https://www.nastygal.com/ca/search?q=orange+top',
'https://www.nastygal.com/ca/search?q=orange%20top&sz=80&start=80',
       'https://www.nastygal.com/ca/search?q=orange%20top&sz=80&start=160'
       'https://www.nastygal.com/ca/search?q=orange%20top&sz=80&start=240'
       'https://www.nastygal.com/ca/search?q=orange%20top&sz=80&start=320'
       'https://www.nastygal.com/ca/search?q=orange%20top&sz=80&start=400']
for i in range(0,len(url)):

    label = search_terms[i//pages]
    label = ' '.join(label)

    print('\nLooking for (',label,')')
    result = requests.get(url[i], headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
    print('Status code:',result.status_code)
    if result.status_code == 200:
        soup = BeautifulSoup(result.content, "html.parser")

    img_urls = []
    imgs = soup.findAll('img', {'itemprop': "image"})
    # print(imgs[1])
    for img in imgs:
        try:
            img_urls += ['https:' + img['content']]
        except:
            pass
    print(img_urls)

    path = './'+folder_name +'/' + store + '/' + label
    print('Path: ',path)

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    # print(num_in_page)
    temp = i//pages+1
    for n, data in enumerate(img_urls):

        # print('# ', (n+sum(num_in_page[0:i+1]))/temp, data)
        try:
            urllib.request.urlretrieve(data,f"{path}/{bigcnt}.jpg")
            bigcnt += 1
        except:
            pass