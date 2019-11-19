import os
import urllib.request
import requests
from bs4 import BeautifulSoup
store = 'hm'
combos = [['turtle', 'neck', 'top'], ['v', 'neck', 'top'],['collared', 'top'],['crew', 'neck', 'top'],['square','neck','top'], ['scoop','neck','top']]
combos = [['round','neck','top']]
# combos = [['long','sleeve','top'],['short','sleeve','top'],['sleeveless','top']]
combos = [["white","top"],["yellow","top"],["green","top"],["orange","top"],["blue","top"],["red","top"]]

# os.mkdir(f'./data/{store}')
urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = f'./data/{store}/'+ fname
    if not os.path.exists(path):
        os.mkdir(path)
    url = "https://www2.hm.com/en_ca/search-results.html?q="

    for c in comb:
        url += c + "+"
    urls += [[url[:-1] + "&page-size=200", fname]]

for perm in urls:
    url = perm[0]
    fname = perm[1]
    for k in range(1,2):
        # url = url[:-1]+ str(k)
        print(url)
        print('search:', fname)
        result = requests.get(url+'#1_0',headers={'User-Agent': 'Mozilla/5.0'})
        if result.status_code == 200:
            soup = BeautifulSoup(result.content, "html.parser")

        pics = []
        imgs = soup.findAll('img', {'class': 'item-image'})
        for img in imgs:
            # print(img)
            try:
                pics += ['https:' + img['src']]
            except:
                pass
            try:
                pics += ['https:' + img['data-altimage']]
            except:
                pass
            # try:
            #     pics += ['https:' + img['data-src']]
            # except:
            #     pass

        print(len(pics))
        for i, u in enumerate(pics):
            urllib.request.urlretrieve(u, f"./data/{store}/{fname}/A_{i}_{k}.jpg")
