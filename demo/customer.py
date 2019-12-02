from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(1, 'demo/customer_helpers')
from customer_helpers import *

""" This file searches the database with your input string and stores the results in an output folder"""
"""******************************How to use this file*****************************************************************************"""
# To search database: do_action = 'search', search_terms = 'INPUT HERE YOUR SEARCH STRING AS 1 ELEMENT (ex. 'black tops turtleneck')
                                          #output_folder = 'WHERE YOU WANT THE IMAGES FOUND FROM THE SEARCH TO BE STORED'
# To clear your output folder: do_action = "clear outputs"
"""*******************************make sure to fill out these parameters *********************************************************"""
do_action = 'blah'
searchterms = 'blah'
output_folder = 'blah'
"""*******************************************************************************************************************************"""

data_folder = './finaltest'
def clear_folder(data_folder):
    names = [data_folder + '/' + f for f in listdir(data_folder) if isfile(join(data_folder, f))]
    print(names)
    for filename in names:
        print(filename)
        os.remove(filename)

def search(L):
    data_folder = "./static/customer/"
    results = get_things(L)
    # print('res',results)
    out = []
    for i in range(0,10):
        try:
            img = Image.open('.'+results[i][1])
            print('i',img)
            img.save(data_folder + str(results[i][0]) + '.jpg',"JPEG", quality=90)
            out += [str(results[i][0]) + '.jpg']
            print(str(results[i][0]))
        except:
            pass
    print(out)
    return out

# delete()
# print(store(data_folder))

def main(do_action, searchterms,output_folder):
    for action in do_action:
        if action == 'search':
            search(searchterms.split())
        elif action == 'clear outputs':
            clear_folder(output_folder)

main(do_action,searchterms, output_folder)





