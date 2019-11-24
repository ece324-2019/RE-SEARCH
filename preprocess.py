from PIL import Image
import os
import glob
import io

root='./webscraping/data'
combos = ['long sleeve top', 'short sleeve top', 'sleeveless top']
combos = ['black top', 'blue top', 'green top', 'orange top', 'red top', 'yellow top','white top']
# combos = ['top with buttons', 'top']
# combos = ['collared top','v neck top','crew neck top','square neck top','turtle neck top','scoop neck top']
combos= ['vneck','turtleneck','crew','square','tops']
stores = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]

h = {'long sleeve top':0,
     'short sleeve top':0,
     'sleeveless top':0}
# h = {'collared top':0,
#      'v neck top':0,
#      'crew neck top':0,
#      'square neck top':0,
#      'turtle neck top':0,
#      'scoop neck top':0}
h={'black top':0,
   'blue top':0,
   'green top':0,
   'orange top':0,
    'red top':0,
   'yellow top':0,
   'white top':0
}

h = {'vneck':0,
     'crew':0,
     'square':0,
     'turtleneck':0,
     'tops':0}
# print(h['sleeveless top'])
for store in stores:
    subroot = root + '/' + store
    folders = [ item for item in os.listdir(subroot) if os.path.isdir(os.path.join(subroot, item)) ]
    for folder in folders:
        # print(folder)
        if folder in combos:
            subfolder = subroot + '/' + folder
            for filename in glob.glob(subfolder+'/*.jpg'):


                try:
                    img = Image.open(open(filename, 'rb'))
                    new_img = img  # .resize((100,100))
                    folder1 = 'finaltest'
                    if not os.path.exists(folder1):
                        os.mkdir(folder1)
                        print('made folder:',filename)
                    newfilename = os.path.join(folder1, store + '_' + folder + str(h[folder]))
                    h[folder] += 1
                    new_img.save(newfilename+'.jpg', "JPEG", quality=90)
                except:
                    print(filename,store)
