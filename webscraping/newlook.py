import os
import urllib.request
import requests
from bs4 import BeautifulSoup
store = 'newlook'
combos = [['turtle', 'neck', 'top'], ['v', 'neck', 'top'],['collared', 'top'],['crew', 'neck', 'top'],['square','neck','top'], ['scoop','neck','top']]
combos = [['long','sleeve','top'],['short','sleeve','top'],['sleeveless','top']]
combos = [["top","with","buttons"]]

# os.mkdir(f'./data/{store}')
urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = f'./data/{store}/'+ fname
    if not os.path.exists(path):
        os.mkdir(path)
    url = "https://www.newlook.com/uk/search/?text="
    for c in comb:
        url += c + "+"
    urls += [[url[:-1] , fname]]

for perm in urls:
    url = perm[0]
    fname = perm[1]
    print('search:', fname)
    result = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
    if result.status_code == 200:
        soup = BeautifulSoup(result.content, "html.parser")

    pics = []
    imgs = soup.findAll('img')
    for img in imgs:
        try:
            a = 'https:' + img['data-srcset']
            pics += [a]
        except:
            continue
    print('found',len(pics),'...')
    for i, u in enumerate(pics):
        try:
            urllib.request.urlretrieve(u, f"./data/{store}/{fname}/A_{i}.jpg")
        except:
            print("BAD", u)
