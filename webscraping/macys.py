import os
import urllib.request
import requests
from bs4 import BeautifulSoup
store = 'macys'
# combos = [['turtle', 'neck', 'top'], ['v', 'neck', 'top'],['collared', 'top'],['crew', 'neck', 'top'],['square','neck','top'], ['scoop','neck','top']]
combos = [['long','sleeve','top'],['short','sleeve','top'],['sleeveless','top']]
combos = [["top","with","buttons"],["top"],["black","top"],["white","top"],["yellow","top"],["green","top"],["orange","top"],["blue","top"],["red","top"]]

# os.mkdir(f'./data/{store}')
urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = f'./data/{store}/'+ fname
    if not os.path.exists(path):
        os.mkdir(path)
    url = "https://www.macys.com/shop/featured/"  # /Productsperpage/120"

    for c in comb:
        url += c + "-"
    urls += [[url[:-1] + "/Pageindex/1", fname]]

for perm in urls:
    url = perm[0]
    fname = perm[1]
    for k in range(1,11):
        url = url[:-1]+ str(k)
        print(url)
        print('search:', fname)
        result = requests.get(url+'#1_0',headers={'User-Agent': 'Mozilla/5.0'})
        if result.status_code == 200:
            soup = BeautifulSoup(result.content, "html.parser")

        pics = []
        imgs = soup.findAll('img', {'class': 'thumbnailImage'})
        for img in imgs:
            try:
                a = img['src']
                # print(a)
                pics += [a]
            except:
                pass

        print(len(pics))
        for i, u in enumerate(pics):
            urllib.request.urlretrieve(u, f"./data/{store}/{fname}/A_{i}_{k}.jpg")
