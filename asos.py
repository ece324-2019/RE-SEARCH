import os
import urllib.request
import requests
from bs4 import BeautifulSoup

combos = [['turtle', 'neck', 'top'], ['v', 'neck', 'top'],['collared', 'top'],['crew', 'neck', 'top'],['square','neck','top']]
combos = [['long','sleeve','top'],['short','sleeve','top'],['sleeveless','top']]
combos = [["top","with","buttons"],["top"],["black","top"],["white","top"],["yellow","top"],["green","top"],["orange","top"],["blue","top"],["red","top"]]

urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = './data/asos2/'+fname
    if not os.path.exists(path):
        os.mkdir(path)
    url = "https://www.asos.com/search/?page=1&q="
    for c in comb:
        url += c + "%20"
    urls += [[url[:-3]+ "&refine=floor:1000" , fname]]
h = {}
for perm in urls:
    url = perm[0]
    for j in range(1, 30):
        url = url[:34] + str(j) + url[35:]
        print(j,url)
        fname = perm[1]
        print('search:', fname)
        result = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if result.status_code == 200:
            soup = BeautifulSoup(result.content, "html.parser")
        urs = []
        imgs = soup.findAll('img',{'data-auto-id':"productTileImage"})
        for img in imgs:
            urs += ['http:'+img["src"]]
        # print(len(urs),'items found ...')
        for i, u in enumerate(urs):
            try:
                opener = urllib.request.URLopener() ## ZAWFUL
                opener.addheader('User-Agent', 'Mozilla/5.0')
                _ = opener.retrieve(u, f"./data/asos2/{fname}/b{i+(j-1)*36}.jpg")
            except:
                print('error',url)
        # print("done saving files")
