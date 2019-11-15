import os
import time
import urllib.request
import requests
from bs4 import BeautifulSoup
store = 'yesstyle'
combos = [['turtle', 'neck', 'top'], ['v', 'neck', 'top'],['collared', 'top'],['crew', 'neck', 'top'],['square','neck','top'], ['scoop','neck','top']]
# combos = [['round','neck','top']]
# combos = [['long','sleeve','top'],['short','sleeve','top'],['sleeveless','top']]
# combos = [["top","with","buttons"],["top"],["black","top"],["white","top"],["yellow","top"],["green","top"],["orange","top"],["blue","top"],["red","top"]]
combos = [['yellow','top']]
# os.mkdir(f'./data/{store}')
urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = f'./data/{store}/'+ fname
    if not os.path.exists(path):
        os.mkdir(path)
    url = "https://www.yesstyle.com/en/list.html?q="
    # "square+neck+top&bpt=48#/ss=120&pn=5&q=square+neck+top&bpt=48&bt=48&s=10&l=1&sb=136"

    for c in comb:
        url += c + "+"
    url = url[:-1] + "&pn=1"
    stop = len(url)-4
    # for c in comb:
    #     url += c + '+'
    # url = url[:-1] + "&bt=48&s=10&l=1&sb=136"
    # print(url)
    urls += [[url, fname, stop]]

for perm in urls:
    url = perm[0]
    fname = perm[1]
    stop = perm[2]
    for k in range(1,2):
        url = url[:-1]+ str(k)
        print(url)
        print('search:', fname)
        result = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        if result.status_code == 200:
            # time.sleep(1)
            soup = BeautifulSoup(result.content, "html.parser")
        pics = []
        imgs = soup.findAll('img')
        # print(len(imgs),imgs[0])
        for img in imgs:
            # print(img)
            try:
                pics += [img['src-lazy']]
            except:
                pass

        print(len(pics))
        for i, u in enumerate(pics):
            try:
                urllib.request.urlretrieve(u, f"./data/{store}/{fname}/A_{i}_{k}.jpg")
            except:
                print(u)
