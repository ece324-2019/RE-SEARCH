import os
import urllib.request
import requests
from bs4 import BeautifulSoup
#['long','sleeve','top'],['short','sleeve','top'],

soup = BeautifulSoup(open('test.html'), 'html.parser')
combos = [['blue','top']]
store = 'netport'
# os.mkdir(f'./data/{store}')
urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = f'./data/{store}/'+ fname
    if not os.path.exists(path):
        os.mkdir(path)

pics = []
# print(soup)
imgs = soup.findAll('div',{'class':"product-image"})
# print(len(imgs),imgs[0])
for img in imgs:
    try:
        img = img.find('img')
        pics += ['https:'+img['data-image-product']]
    except:
        pass

print(len(pics))
for i, u in enumerate(pics):
    urllib.request.urlretrieve(u, f"./data/{store}/{fname}/C_{i}.jpg")
