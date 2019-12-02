from test import *
from PIL import Image
import os
import uuid
import glob
from integrate import integration
from os import listdir
from os.path import isfile, join

""" This files takes all the images in a folder, runs them through our model and then stores the results in a local database"""
"""******************************How to use this file*****************************************************************************"""
# To populate database: do_action = 'store, data_folder = 'THE LOCATION OF THE IMAGES YOU WANT TO CLASSIFY AND STORE INTO THE DATABASE'
# To clear the database: do_action = 'clear database'
"""*******************************make sure to fill out these parameters *********************************************************"""
do_action = 'blah'
data_folder = 'blah'
"""*******************************************************************************************************************************"""

def clear_folder(data_folder):
    names = [data_folder + '/' + f for f in listdir(data_folder) if isfile(join(data_folder, f))]
    print(names)
    for filename in names:
        print(filename)
        os.remove(filename)
def store(data_folder):
    """ go inside data_folder and get all the names of the images"""
    names = []
    names= [data_folder +'/' + f for f in listdir(data_folder) if isfile(join(data_folder, f))]
    # print(names)
    """ get labels"""
    out_c = integration(data_folder, 'colors')
    out_s = integration(data_folder, 'sleeves')
    out_n = integration(data_folder, 'necklines')
    out_b = integration(data_folder, 'buttons')

    addparams = ['UUID','color','color_confidence','neckline','neckline_confidence','sleeves','sleeves_confidence','buttons','buttons_confidence','path_to_file']
    for i in range(0,len(out_c)):
        addlist = [[str(uuid.uuid4()),str(out_c[i][0]),str(out_c[i][1]),str(out_n[i][0]),str(out_n[i][1]),str(out_s[i][0]),str(out_s[i][1]),str(out_b[i][0]),str(out_b[i][1]),str(names[i])]]
        write(addparams, addlist)

def main(do_action, data_folder):
    for action in do_action:
        if action == 'store':
            store(data_folder)
        elif action == 'clear database':
            delete()

main(do_action, data_folder)