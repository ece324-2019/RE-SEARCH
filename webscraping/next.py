#https://www3.next.co.uk/search?w=orange%20long%20sleeve%20top&isort=score&af=

import urllib.request
import requests
from bs4 import BeautifulSoup
import os

""" INPUT PARAMETERS """
store = 'Next'
concat = "%20"

scroll = True
num_tops = 380
per_page= 24
#search_terms = [["top","with","buttons"],["top"],["black","top"],["white","top"],["yellow","top"],["green","top"],["orange","top"],["blue","top"],["red","top"]]
search_terms = [["button",'top']]
"""*****************"""
if scroll == False:
    pages = 1
else:
    pages = num_tops//per_page + 1
print('# Pages: ',pages)

#"https://www.tobi.com/ca/search?page=2&search_term="
#https://www.tobi.com/ca/search?search_term=top%20women

if scroll == False:
    url_base = "https://www3.next.co.uk/search?w="
    add = "&isort=score&af="
else:
    first = "https://www3.next.co.uk/search?w="
    #https://www3.next.co.uk/search?w=orange%20long%20sleeve%20top&isort=score&af=&srt=24
    url_base = "https://www3.next.co.uk/search?w="
    add = "&isort=score&af=&"

if os.path.exists('./scraped/'+store):
    pass
else:
    os.mkdir('./scraped/'+store)

print('Store: ',store)
url = []

if scroll == False:
    for i in range(0,len(search_terms)):
        temp = url_base
        for j in range(0,len(search_terms[i])-1):
            temp = temp + search_terms[i][j] + concat
        url += [temp + search_terms[i][len(search_terms[i])-1] + add]
        print('# ',i, ': ',url[len(url)-1])
else:
    for i in range(0,len(search_terms)):
        for k in range(0,pages):
            temp = url_base
            if k == 0:
                for j in range(0, len(search_terms[i]) - 1):
                    temp = temp + search_terms[i][j] + concat
                url += [temp + search_terms[i][len(search_terms[i]) - 1] + add]
                print('# ', i, ': ', url[len(url) - 1])
            else:
                for j in range(0,len(search_terms[i])-1):
                    temp = temp + search_terms[i][j] + concat
                url += [temp + search_terms[i][len(search_terms[i])-1]  + add + 'srt=' + str(k*24)]
                print('# ',i, ': ',url[len(url)-1])

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
    imgs = soup.findAll('div', {'class': 'Images'})
    for img in imgs:
        try:
            img_urls += [img.find('img')['src']]
            # print('Image Found: ', img.find('img')['src'])
        except:
            pass


    path = './scraped/' + store + '/' + label
    print('Path: ',path)

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    curr_page = i%pages
    for n, data in enumerate(img_urls):
        urllib.request.urlretrieve(data,f"{path}/{n+curr_page*per_page}.jpg")

