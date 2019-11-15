import os
import urllib.request
import requests
from bs4 import BeautifulSoup
store = 'shopbop'
# combos = [['turtle', 'neck', 'top'], ['v', 'neck', 'top'],['long','sleeve','top'],['collared', 'top'],['crew', 'neck', 'top'],['square','neck','top'], ['scoop','neck','top']]
combos = [['short','sleeve','top']]#,['sleeveless','top']]
# combos = [["top","with","buttons"],["top"],["black","top"],["white","top"],["yellow","top"],["green","top"],["orange","top"],["blue","top"],["red","top"]]
combos = [['long','sleeve','top']]
combos = [['red','shirt']]
# os.mkdir(f'./data/{store}')
urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = f'./data/{store}/'+ fname
    if not os.path.exists(path):
        os.mkdir(path)
    url = "https://www.shopbop.com/products?query="  # /Productsperpage/120"

    for c in comb:
        url += c + "+"
    urls += [[url[:-1] + "&baseIndex=000", fname]]

for perm in urls:
    url = perm[0]
    fname = perm[1]
    for k in range(0,1):
        if k >0:
            url = url[:-3]+ str(k*100)
        print(url)
        print('search:', fname)
        result = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        if result.status_code == 200:
            soup = BeautifulSoup(result.content, "html.parser")

        pics = []
        imgs = soup.findAll('div', {'data-at': 'productContainer'})
        # print(len(imgs), imgs[0])
        for img in imgs:
            try:
                pic = img.find('img')
                pics += [pic['src']]
            except:
                pass

        print(len(pics))
        for i, u in enumerate(pics):
            urllib.request.urlretrieve(u, f"./data/{store}/{fname}/A_{i}_{k}.jpg")
