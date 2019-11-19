import urllib.request
import requests
from bs4 import BeautifulSoup
import os
import numpy as np


url = "https://www.net-a-porter.com/Shop/Search/sleeveless+top?npp=view_all&keywords=sleeveless%20top&fbclid=IwAR2N0FMOKPepjU1CbwIdEU_tBN0j4DU35MxbrwuM2wCrPXzyvWa9pUp8lXI"
result = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
print(result)