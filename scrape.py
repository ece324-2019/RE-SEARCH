import urllib.request
import requests
from bs4 import BeautifulSoup
import json

# url = "https://www.aritzia.com/en/search?q=black%20long%20sleeve%20top&sz=100"
# url = "https://www.asos.com/search/?q=black+long+sleeve+square+top"
# url = "https://www.zaful.com/s/red-v-neck-top/"
# url = "https://ca.boohoo.com/search?q=yellow+v+neck+long+sleeve"
# url = "https://www.fashionnova.com/pages/search-results?q=turtle%20neck%20top&src=isp&page_num=1"
# url = "https://www.next.co.uk/search?w=turtle%20neck%20top&af=gender:women&isort=score#1_0"
url = "https://www.newlook.com/uk/search/?text=turtle+neck+top"
url = "https://www.prettylittlething.us/catalogsearch/result/?q=turtle+neck+top&page=2&ajax=0"
url = "https://www.riverisland.com/search?keyword=turtle%20neck%20top&search-submit=&f-division=women"
url = "https://lulus.com"
url = "https://www.macys.com/shop/featured/turtle-neck-top/Pageindex/3"#/Productsperpage/120"
result = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
print(result.status_code)

if result.status_code == 200:
    soup = BeautifulSoup(result.content, 'html.parser')

""" ARITZIA"""
# urls = []
# imgs = soup.findAll('div',{'class':'product-image js-product-plp-image'})
# for img in imgs:
#     urls += [img.find('img',{'class':'lazy'})["data-original"]]
# print(len(urls))

""" ASOS """
# urls = []
# imgs = soup.findAll('img',{'data-auto-id':"productTileImage"})
# for img in imgs:
#     urls += ['http:'+img["src"]]
# print(len(urls))

"""ZAFUL"""
urls = []
# imgs = soup.findAll('div',{'class':"img-hover-wrap"})
# for img in imgs:
#     urls += [img.find('img')["data-original"]]

""" BOO HOO"""
# urls = []
# from ast import literal_eval
#
# imgs = soup.findAll('img',{'class':"swatch-image js-required js-product-tile-swatch lazyload"})
# for img in imgs:
#     img = literal_eval(img["data-thumb"])
#     urls += ['https:'+img["src"]]
# print(len(urls))

""" MISGUIDED """
# pics = []
# imgs = soup.findAll('div',{'class':'Images'})
# for img in imgs:
#     pics += [img.find('img')['src']]
# print(len(pics))

""" NEW LOOK """
# pics = []
# imgs = soup.findAll('div',{'class':'category-product js-productitem js-select-productimpression'})
# for img in imgs:
#     try:
#         a = img.find('img')['src']
#         print(a)
#         pics += [a]
#     except:
#         print(img)
# print(len(pics))

""" RIVER ISLAND"""
# pics = []
# imgs = soup.findAll('img',{'data-qa':'product-image'})
# for img in imgs:
#     try:
#         a = img['src']
#         print(a)
#         pics += [a]
#     except:
#         print(img)
# print(len(pics))
""" MACYS """
pics = []
imgs = soup.findAll('img',{'class':'thumbnailImage'})
for img in imgs:
    try:
        a = img['src']
        print(a)
        pics += [a]
    except:
        print(img)
print(len(pics))