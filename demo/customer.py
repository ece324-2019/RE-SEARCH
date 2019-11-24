from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(1, 'demo/customer_helpers')
from customer_helpers import *

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







