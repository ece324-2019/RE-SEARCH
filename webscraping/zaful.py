import os
import urllib.request
import requests
from bs4 import BeautifulSoup
url = "https://www.zaful.com/s/red-v-neck-top/"


combos = [['v', 'neck', 'top'],['crew', 'neck', 'top']]
# combos = [['square','neck','top']]
# combos = [['long','sleeve','top'],['short','sleeve','top']]
# combos = [['sleeveless','top']]
# combos = [["top","with","buttons"]]

urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = './data/zaful/'+fname
    # os.mkdir('./data/zaful')
    if not os.path.exists(path):
        os.mkdir(path)
    url = "https://www.zaful.com/s/"
    for c in comb:
        url += c + "-"
    urls += [[url[:-1]+ "/g_6.html" , fname]]
# print(urls[0][0][:-6]+ str(6) + urls[0][0][-5:])
h = {}
for perm in urls:
    url = perm[0]
    fname = perm[1]
    for i in range(1,2):
        # url = url[:-6] + str(i) + url[-5:]
        print('search:', fname)
        result = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if result.status_code == 200:
            soup = BeautifulSoup(result.content, "html.parser")
        urs = []
        imgs = soup.findAll('div', {'class': "img-hover-wrap"})
        for img in imgs:
            urs += [img.find('img')["data-original"]]

        print(len(urs),'items found ...')
        for j, url in enumerate(urs):
            try:
                opener = urllib.request.URLopener() ## ZAWFUL
                opener.addheader('User-Agent', 'Mozilla/5.0')
                _ = opener.retrieve(url, f"./data/zaful/{fname}/f{j+120*(i-1)}.jpg")
            except:
                print(url)
        print("done saving files")


