
import urllib.request
import requests
from bs4 import BeautifulSoup
import os

""" INPUT PARAMETERS """
store = 'Modcloth'
folder_name = 'scraped'

num_per_page = 300
scroll = True
num_tops = 300

search_terms = [['long','sleeve']]
#,["top"],["black","top"],["white","top"],["yellow","top"],["green","top"],["orange","top"],["blue","top"],["red","top"],["pattern","top"]]
"""*****************"""

pages = (num_tops-1)//num_per_page + 1

if scroll == False:
    url_base = "https://www.ssense.com/en-ca/women?q=v-neck&page=1"
else:
    url_base = "https://www.modcloth.com/search?q="
    add = "&prefn1=productLevel&sz=300&prefv1=variationgroup"

if os.path.exists('./'+folder_name +'/'+store):
    pass
else:
    os.mkdir('./'+folder_name +'/'+store)

print('Store: ',store)
url = []

if scroll == False:
    for i in range(0,len(search_terms)):
        temp = url_base
        for j in range(0,len(search_terms[i])-1):
            temp = temp + search_terms[i][j] + '%20'
        url += [temp + search_terms[i][len(search_terms[i])-1] ]
        print('# ',i, ': ',temp + search_terms[i][len(search_terms[i])-1])
else:
    for i in range(0,len(search_terms)):
        for k in range(1,pages+1):
            temp = url_base

            for j in range(0,len(search_terms[i])-1):
                temp = temp + search_terms[i][j] + '%20'
            url += [temp + search_terms[i][len(search_terms[i])-1] + add]
            print('# ',i, ': ',temp + search_terms[i][len(search_terms[i])-1] + add)


tally = []
url = ['https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=1&prefv1=variationgroup',
'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=101&prefv1=variationgroup',
'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=151&prefv1=variationgroup',
'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=201&prefv1=variationgroup',
'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=251&prefv1=variationgroup',
'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=301&prefv1=variationgroup',
'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=351&prefv1=variationgroup',
'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=401&prefv1=variationgroup',
       'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=451&prefv1=variationgroup',
       'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=501&prefv1=variationgroup',
       'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=551&prefv1=variationgroup',
       'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=601&prefv1=variationgroup',
       'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=651&prefv1=variationgroup',
       'https://www.modcloth.com/search?q=sleeveless&prefn1=productLevel&sz=51&start=701&prefv1=variationgroup',
       ]


bigcnt = 0
for i in range(0,len(url)):

    # label = search_terms[i//pages]
    # label = ' '.join(label)
    label = 'sleeveless'
    print('\nLooking for (',label,')')
    result = requests.get(url[i], headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
    print('Status code:',result.status_code)
    if result.status_code == 200:
        soup = BeautifulSoup(result.content, "html.parser")

    img_urls = []
    imgs = soup.findAll('ul', {'class': 'swatch-list'})
    # print(imgs)
    for img in imgs:
        try:
            img_urls += [img.find('picture')['data-thumb-main']]
        except:
            pass

    path = './'+folder_name +'/' + store + '/' + label
    print('Path: ',path)

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    curr_page = i%pages
    for n, data in enumerate(img_urls):
        # print('# ', n+curr_page*num_per_page, data)
        try:
            urllib.request.urlretrieve(data,f"{path}/{bigcnt}.jpg")
            bigcnt += 1
        except:
            pass