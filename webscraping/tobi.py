import urllib.request
import requests
from bs4 import BeautifulSoup
import os

""" INPUT PARAMETERS """
store = 'Tobi'
folder_name = 'scraped'

per_page =
scroll = True
num_tops = 100

search_terms = [["top","with","buttons"],["top"],["black","top"],["white","top"],["yellow","top"],["green","top"],["orange","top"],["blue","top"],["red","top"],["pattern","top"]]
"""*****************"""

pages = num_tops//95 + 1
#"https://www.tobi.com/ca/search?page=2&search_term="
#https://www.tobi.com/ca/search?search_term=top%20women
if scroll == False:
    url_base = "https://www.tobi.com/ca/search?search_term="
else:
    url_base = "https://www.tobi.com/ca/search?"
    add = "&search_term="

if os.path.exists('./'+folder_name +'/'+store):
    pass
else:
    os.mkdir('./'+folder_name +'/'+store)

print('Store: ',store)
url = []

if scroll == False:
    for i in range(0,len(search_terms)):
        temp = url_base
        for j in range(0,len(search_terms[i])-1):
            temp = temp + search_terms[i][j] + '%20'
        url += [temp + search_terms[i][len(search_terms[i])-1]]
        print('# ',i, ': ',temp + search_terms[i][len(search_terms[i])-1])
else:
    for i in range(0,len(search_terms)):
        for k in range(1,pages+1):
            if k == 1:
                temp = url_base + 'q='
            else:
                temp = url_base + 'page=' + str(k) + '%20'

            for j in range(0,len(search_terms[i])-1):
                temp = temp + search_terms[i][j] + '+'
            url += [temp + search_terms[i][len(search_terms[i])-1]]
            print('# ',i, ': ',temp + search_terms[i][len(search_terms[i])-1])

#url = "https://www.scotch-soda.com/ca/en/search?q=patterned+top+women&go=go"

tally = []
for i in range(0,len(url)):

    label = search_terms[i//pages]
    label = ' '.join(label)

    print('\nLooking for (',label,')')
    result = requests.get(url[i], headers={'User-Agent': 'Mozilla/5.0'})
    print('Status code:',result.status_code)
    if result.status_code == 200:
        soup = BeautifulSoup(result.content, "html.parser")

    img_urls = []
    imgs = soup.findAll('div',{'class':"product-images"})
    for img in imgs:
        try:
            img_urls += [img.find('img')['data-lazy-src']]
            # print('Image Found: ', img.find('img')['data-lazy-src'])
        except:
            pass


    path = './'+folder_name +'/' + store + '/' + label
    print('Path: ',path)

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    curr_page = i%pages
    for n, data in enumerate(img_urls):
        urllib.request.urlretrieve(data,f"{path}/{n+curr_page*95}.jpg")

# print("\nSummary")
# print("Store: ",store,'\n')
# for i in range(0,len(tally)):
#     print(tally[i][0],tally[i][1])
