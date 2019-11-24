import os
import time
import urllib.request
import requests
from bs4 import BeautifulSoup
store = 'kohls1'
# combos = [['button', 'top']]#, ['v', 'neck', 'shirt'],['collared', 'shirt'],['crew', 'neck', 'shirt'],['square','neck','shirt']]
# os.mkdir(f'./data/{store}')
urls = [['https://www.kohls.com/catalog/womens-cowlneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:Turtleneck+Neckline:Mockneck+Neckline:Cowlneck+Category:Tops+Department:Clothing&BL=y&icid=genturtlenecks-VN1-womens&pfm=search%20p13n_no%20Visual%20Nav&kls_sbp=30202820972503232343190607550603629498&PPP=120&S=1',
         'turtleneck'],['https://www.kohls.com/catalog/womens-vneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:V-Neck+Category:Tops+Department:Clothing&S=1&PPP=120&pfm=search%20visual%20nav%20refine&kls_sbp=30202820972503232343190607550603629498','vneck'],
        ['https://www.kohls.com/catalog/womens-vneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:V-Neck+Category:Tops+Department:Clothing&S=2&PPP=120&pfm=search%20visual%20nav%20refine&kls_sbp=30202820972503232343190607550603629498','vneck'],
        ['https://www.kohls.com/catalog/womens-squareneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:Squareneck+Category:Tops+Department:Clothing&S=2&PPP=120&pfm=search%20visual%20nav%20refine&kls_sbp=30202820972503232343190607550603629498','square'],
        ['https://www.kohls.com/catalog/womens-white-crewneck-tops-clothing.jsp?CN=Gender:Womens+Color:Black+Color:Blue+Color:Orange+Color:Green+Color:Red+Color:Yellow+Color:White+Neckline:Crewneck+Category:Tops+Department:Clothing&BL=y&S=2&PPP=120&pfm=search%20visual%20nav%20refine&kls_sbp=30202820972503232343190607550603629498','crew'],
        ['https://www.kohls.com/catalog/womens-cowlneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:Turtleneck+Neckline:Mockneck+Neckline:Cowlneck+Category:Tops+Department:Clothing&BL=y&icid=genturtlenecks-VN1-womens&pfm=search%20p13n_no%20Visual%20Nav&kls_sbp=30202820972503232343190607550603629498&PPP=120&S=2','turtleneck'],
        ['https://www.kohls.com/catalog/womens-henley-tops-clothing.jsp?CN=Gender:Womens+Neckline:Henley+Category:Tops+Department:Clothing&S=2&PPP=120&pfm=search%20visual%20nav%20refine&kls_sbp=30202820972503232343190607550603629498','vneck']]
# for comb in combos:
#     comb = ' '.join(comb).split()
#     fname = ' '.join(comb)
#     path = f'./data/{store}/'+ fname
#     if not os.path.exists(path):
#         os.mkdir(path)
#     url = "https://www.emmacloth.com/pdsearch/page1/"
#     for c in comb:
#         url += c + "-"
#     url = url[:-1] + "/"
#     stop = len(url)-4
#     url = 'https://www.emmacloth.com/pdsearch/page11/button-top/'
#     urls += [[url, fname, stop]]
cnt = 0
for perm in urls:
    url = perm[0]
    fname = perm[1]
    for k in range(1,2):
        print(url)
        print('search:', fname)
        result = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        if result.status_code == 200:
            soup = BeautifulSoup(result.content, "html.parser")
        pics = []
        imgs = soup.findAll('img', {'class': 'pmp-hero-img'})
        for img in imgs:
            try:
                # img = img.find('img')
                pics += [img['data-herosrc']]
            except:
                pass
        print(len(pics))
        for i, u in enumerate(pics):
            try:
                cnt += 1
                # opener = urllib.request.build_opener()
                # opener.addheaders = [('User-Agent',
                #                       'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
                # urllib.request.install_opener(opener)
                urllib.request.urlretrieve(u, f"./data/{store}/{fname}/A_{cnt}.png")
            except:
                print(u)
