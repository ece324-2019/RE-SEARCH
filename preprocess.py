from PIL import Image
import os
import glob

root='./webscraping/data'
combos = ['long sleeve top', 'short sleeve top', 'sleeveless top']
# combos = ['black top', 'blue top', 'green top', 'orange top', 'red top', 'yellow','white']
# combos = ['top with buttons', 'top']
# combos = ['collared top','v neck top','crew neck top','square neck top','turtle neck top','scoop neck top']

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

# print(h['sleeveless top'])
for store in stores:
    subroot = root + '/' + store
    folders = [ item for item in os.listdir(subroot) if os.path.isdir(os.path.join(subroot, item)) ]
    for folder in folders:
        # print(folder)
        if folder in combos:
            subfolder = subroot + '/' + folder
            for filename in glob.glob(subfolder+'/*.jpg'):
                if not os.path.exists(folder):
                    os.mkdir(folder)
                    print('made folder:',filename)
                img = Image.open(filename)
                new_img = img.resize((100,100))
                newfilename = os.path.join(folder, store + '_' +folder+ str(h[folder]))
                h[folder] += 1
                try:
                    new_img.save(newfilename+'.jpg', "JPEG", quality=90)
                except:
                    print(filename,store)
