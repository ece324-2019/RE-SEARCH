from test import *
from PIL import Image
import os
import uuid
import glob
from integrate import integration
from os import listdir
from os.path import isfile, join

# first process all the photos in the database

# run it through the model

data_folder = './finaltest'
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
    # print(out_c)
    for i in range(0,len(out_c)):
        # print(uuid.uuid4())
        # addlist = [[str(uuid.uuid4())[0:9], str(out_c[i][0]), str(out_c[i][1]),'NULL','0',str(out_s[i][0]), str(out_s[i][1]),'NULL','0',str(names[i])]]
        # addlist = [[str(uuid.uuid4())[0:9], str(out_c[i][0]), str(out_c[i][1]), 'NULL', 'NULL',
        #             'NULL', 'NULL', 'NULL', 'NULL', str(names[i])]]
        # print(out_c[i], out_n[i], out_b[i], out_s[i])
        addlist = [[str(uuid.uuid4()),str(out_c[i][0]),str(out_c[i][1]),str(out_n[i][0]),str(out_n[i][1]),str(out_s[i][0]),str(out_s[i][1]),str(out_b[i][0]),str(out_b[i][1]),str(names[i])]]
        write(addparams, addlist)
        # print(addlist)


delete()
print(store(data_folder))
# clear_folder('../output_images')
# search(['green'])