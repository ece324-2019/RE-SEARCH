import os
import time
import urllib.request
import requests
from bs4 import BeautifulSoup
store = 'rue'
# combos = [ ['v', 'neck', 'tops'],['turtleneck', 'tops'],['crew', 'neck', 'tops'],['square','neck','tops']]
# os.mkdir(f'./data/{store}')
# urls = [['https://www.kohls.com/catalog/womens-cowlneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:Turtleneck+Neckline:Mockneck+Neckline:Cowlneck+Category:Tops+Department:Clothing&BL=y&icid=genturtlenecks-VN1-womens&pfm=search%20p13n_no%20Visual%20Nav&kls_sbp=30202820972503232343190607550603629498&PPP=120&S=1',
#          'turtleneck'],['https://www.kohls.com/catalog/womens-vneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:V-Neck+Category:Tops+Department:Clothing&S=1&PPP=120&pfm=search%20visual%20nav%20refine&kls_sbp=30202820972503232343190607550603629498','vneck'],
#         ['https://www.kohls.com/catalog/womens-vneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:V-Neck+Category:Tops+Department:Clothing&S=2&PPP=120&pfm=search%20visual%20nav%20refine&kls_sbp=30202820972503232343190607550603629498','vneck'],
#         ['https://www.kohls.com/catalog/womens-squareneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:Squareneck+Category:Tops+Department:Clothing&S=2&PPP=120&pfm=search%20visual%20nav%20refine&kls_sbp=30202820972503232343190607550603629498','square'],
#         ['https://www.kohls.com/catalog/womens-white-crewneck-tops-clothing.jsp?CN=Gender:Womens+Color:Black+Color:Blue+Color:Orange+Color:Green+Color:Red+Color:Yellow+Color:White+Neckline:Crewneck+Category:Tops+Department:Clothing&BL=y&S=2&PPP=120&pfm=search%20visual%20nav%20refine&kls_sbp=30202820972503232343190607550603629498','crew'],
#         ['https://www.kohls.com/catalog/womens-cowlneck-tops-clothing.jsp?CN=Gender:Womens+Neckline:Turtleneck+Neckline:Mockneck+Neckline:Cowlneck+Category:Tops+Department:Clothing&BL=y&icid=genturtlenecks-VN1-womens&pfm=search%20p13n_no%20Visual%20Nav&kls_sbp=30202820972503232343190607550603629498&PPP=120&S=2','turtleneck']]
urls = [['https://www.rue21.com/store/girls/tops/yellow-tops/_/N-9qbZ1z140w0?No=32','tops'],
        ['https://www.rue21.com/store/girls/tops/orange-red-tops/_/N-9qbZ1z140w5Z1z140wk?No=32','tops'],
        ['https://www.rue21.com/store/girls/tops/blue-tops/_/N-9qbZ1z140wi?No=32','tops'],
        ['https://www.rue21.com/store/girls/tops/green-tops/_/N-9qbZ1z140wj?No=32','tops'],
        ['https://www.rue21.com/store/girls/tops/white-tops/_/N-9qbZ1z140wg?No=32','tops'],
        ['https://www.rue21.com/store/girls/tops/black-tops/_/N-9qbZ1z140wo?No=32','tops']]
# urls = [['https://www.rue21.com/store/girls/tops/yellow-tops/_/N-9qbZ1z140w0?No=32','stops']]
        # ['https://www.rue21.com/store/girls/tops/yellow-tops/_/N-9qbZ1z140w0Z1z13z15','tops'],
        # ['https://www.rue21.com/store/girls/tops/yellow-tops/_/N-9qbZ1z140w0Z1z13zcu','tops']]#,
        # ['https://www.rue21.com/store/girls/tops/green-tops/_/N-9qbZ1z140wj#page-2','tops'],
        # ['https://www.rue21.com/store/girls/tops/white-tops/_/N-9qbZ1z140wg#page-2','tops'],
        # ['https://www.rue21.com/store/girls/tops/black-tops/_/N-9qbZ1z140wo#page-2','tops']]
# urls = [['gge','dsfd']]
combos = [['stops']]
for comb in combos:
    comb = ' '.join(comb).split()
    fname = ' '.join(comb)
    # path = f'./data/{store}/'+ fname
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # url = "https://www.yoursclothing.co.uk/search?q="
    # for c in comb:
    #     url += c + "+"
    # url = url[:-1] #+ "/"
    # stop = len(url)-4
    # # url = 'https://www.emmacloth.com/pdsearch/page11/button-top/'
    # urls += [[url, fname, stop]]
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
        # soup = BeautifulSoup(open('test.html'), 'html.parser')
        pics = []
        imgs = soup.findAll('img',{'class':'product-link'})
        # print(len(imgs), imgs[100])
        for img in imgs:
            try:
                # img = img.find('img')
                pics += [img['src']]
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
                urllib.request.urlretrieve(u, f"./data/{store}/{fname}/A_{cnt}_.jpg")
            except:
                print(u)
