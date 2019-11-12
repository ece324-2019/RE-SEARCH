import os
import urllib.request
import requests
from bs4 import BeautifulSoup
from ast import literal_eval

url = "https://ca.boohoo.com/search?q=yellow+v+neck+long+sleeve"
combos = [['turtle', 'neck', 'top'], ['v', 'neck', 'top'],['collared', 'top'],['crew', 'neck', 'top'],['square','neck','top']]
# os.mkdir('./data/boohoo')
combos = [["top","with","buttons"],["top"]]
combos = [['long','sleeve','top'],['short','sleeve','top'],['sleeveless','top']]

urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = './data/boohoo/'+ fname
    if not os.path.exists(path):
        os.mkdir(path)
    url = "https://ca.boohoo.com/search?q="
    for c in comb:
        url += c + "%20"
    urls += [[url[:-3] + "&sz=100", fname]]
    urls += [[url[:-3] + "&sz=100&start=100", fname]]


for perm in urls:
    url = perm[0]
    fname = perm[1]
    print('search:', fname)
    result = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
    if result.status_code == 200:
        soup = BeautifulSoup(result.content, "html.parser")
    urs = []
    imgs = soup.findAll('img', {'class': "swatch-image js-required js-product-tile-swatch lazyload"})
    for img in imgs:
        img = literal_eval(img["data-thumb"])
        urs += ['https:' + img["src"]]
    print('found:',len(urs),'...')
    for i, u in enumerate(urs):
        urllib.request.urlretrieve(u, f"./data/boohoo/{fname}/A_{i}.jpg")


#
#
#
# urls = []
# from ast import literal_eval
#

# print(len(urls))