import os
import urllib.request
import requests
from bs4 import BeautifulSoup

# combos = [['long','sleeve','top'],['short','sleeve','top'],['sleeveless','top']]
combos = [["top","with","buttons"],["top"],["black","top"],["white","top"],["yellow","top"],["green","top"],["orange","top"],["blue","top"],["red","top"]]

urls = []
# os.mkdir('./data/aritzia2')
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = './data/aritzia2/'+ fname
    if not os.path.exists(path):
        os.mkdir(path)
    url = "https://www.aritzia.com/en/search?q="
    for c in comb:
        url += c + "%20"
    urls += [[url[:-3] + "&sz=250", fname]]
h = {}
for perm in urls:
    url = perm[0]
    fname = perm[1]
    print('search:', fname)
    result = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
    # print(result.status_code)
    if result.status_code == 200:
        soup = BeautifulSoup(result.content, "html.parser")
    urls = []
    imgs = soup.findAll('div',{'class':'product-image js-product-plp-image'})
    for img in imgs:
        urls += [img.find('img',{'class':'lazy'})["data-original"]]
    print(len(urls),'items found ...')
    # print(urls)
    h[fname] = len(urls)
    for i, url in enumerate(urls):
        urllib.request.urlretrieve(url, f"./data/aritzia2/{fname}/A_{i}.jpg")
    print("done saving files")
import pprint
pprint.pprint(h)

