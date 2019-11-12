
import urllib.request
import requests
from bs4 import BeautifulSoup
import os

""" INPUT PARAMETERS """
store = 'Modcloth'
folder_name = 'data'

num_per_page = 300
scroll = True
num_tops = 300

# search_terms = [["top","with","buttons"]]
search_terms = [['turtleneck', 'top'], ['v-neck', 'top'],['collared', 'top'],['crew', 'neck', 'top'],['square','neck','top'], ['scoop','neck','top']]
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
for i in range(0,len(url)):

    label = search_terms[i//pages]
    label = ' '.join(label)

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
        print('# ', n+curr_page*num_per_page, data)
        try:
            urllib.request.urlretrieve(data,f"{path}/{n+curr_page*num_per_page}.jpg")
        except:
            pass