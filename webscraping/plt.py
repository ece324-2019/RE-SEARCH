import os
import urllib.request
import requests
from bs4 import BeautifulSoup
store = 'PLT'
combos = [['turtle', 'neck', 'top'], ['v', 'neck', 'top'],['collared', 'top'],['crew', 'neck', 'top'],['square','neck','top'], ['scoop','neck','top']]
combos = [["top","with","buttons"]]

# os.mkdir(f'./data/{store}')
urls = []
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    path = f'./data/{store}/'+ fname
    if not os.path.exists(path):
        os.mkdir(path)
        # "https://www.next.co.uk/search?w=orange%20long%20sleeve%20top&isort=score&af=&srt=72"
    # url = "https://www.next.co.uk/search?w="
    url = "https://www.prettylittlething.us/catalogsearch/result/?q="

    for c in comb:
        url += c + "%20"
    urls += [[url[:-3] + "&page=0", fname]]

for perm in urls:
    url = perm[0]
    fname = perm[1]
    for k in range(0,1):
        url = url[:-2]+ str(k*24)
        print(url)
        print('search:', fname)
        result = requests.get(url+'#1_0',headers={'User-Agent': 'Mozilla/5.0'})
        if result.status_code == 200:
            soup = BeautifulSoup(result.content, "html.parser")

        pics = []
        imgs = soup.findAll('div', {'class': 'category-product js-productitem js-select-productimpression'})
        for img in imgs:
            try:
                a = img.find('img')['src']
                # print(a)
                pics += [a]
            except:
                continue
                # print(img)
        print(len(pics))

        for i, u in enumerate(pics):
            urllib.request.urlretrieve(u, f"./data/{store}/{fname}/A_{i}_{k}.jpg")
