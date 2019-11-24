from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os
from flask import Flask, redirect, render_template, request, session, url_for
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from torchvision import transforms
import torch
import sys
import time
sys.path.insert(1, 'demo/customer.py')
from customer import *
# create the application object
app = Flask(__name__)
dropzone = Dropzone(app)
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'manager_results'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/static/manager'
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'
app.config['DROPZONE_DEFAULT_MESSAGE'] = "Upload some pics here !"
app.config['DROPZONE_MAX_FILES'] = 10
app.config['DROPZONE_PARALLEL_UPLOADS'] = 10
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB
# torch.nn.Module.dump_patches = True
""" ML VARIABLES"""
class_c = ["black", "blue", "green", "orange", "red", "white", "yellow"]
class_n = ["crew", "square", "turtle", "v-neck"]
class_s = ["long", "short", "sleeveless"]
class_b = ["no_buttons", "buttons"]
trans_c = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.7, 0.7, 0.7), (0.295, 0.295, 0.295))])
trans_s = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.7, 0.7, 0.7), (0.285, 0.285, 0.285))])
trans_n = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.68, 0.68, 0.68), (0.283, 0.283, 0.283))])
trans_b = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.66, 0.66, 0.66), (0.31, 0.31, 0.31))])
model_c = torch.load('models/model_c1.pt')
model_c.eval()
model_s = torch.load('models/model_s.pt')
model_s.eval()
model_b=torch.load('models/model_0_0.795.pt')
model_b.eval()
print(model_b.conv_bn1)


# model_n = torch.load('model_n.pt')
# model_n.eval()

@app.route('/')
def home():
    return render_template('home.html')  # render a template

@app.route('/customer', methods=['POST','GET'])
def customer():
    if request.method == "POST":
        search_keyword = request.form['text']
        search_keyword = search_keyword.lower().split()
        clear_folder('./static/customer')
        customer_results = search(search_keyword)
        session['customer_results']=customer_results
        return redirect(url_for('customer_results'))
    return render_template('customer.html')  # render a template

@app.route('/customer/results')
def customer_results():
    filenames = []
    if 'customer_results' in session:
        filenames =session['customer_results']
    filenames2 = []
    filenames1 = filenames
    bad_result = ''
    if not filenames:
        bad_result = " Sorry, No Results Found"
    if len(filenames) > 5:
        filenames1 = filenames[:5]
        filenames2 = filenames[5:]
    print(filenames1)
    print(filenames2)
    session.pop('customer_results',None)
    return render_template('customer_results.html',
                           filenames1=filenames1,
                           filenames2=filenames2,
                           bad_result=bad_result)


@app.route('/manager', methods=['GET', 'POST'])
def manager():
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
        session['nn_colours'] = []
        session['nn_necklines'] = []
        session['nn_sleeves'] = []
        # session['nn_buttons'] = []

    # list to hold our uploaded image urls
    file_urls = session['file_urls']
    nn_colours = session['nn_colours']
    nn_necklines = session['nn_necklines']
    nn_sleeves = session['nn_sleeves']
    # nn_buttons = session['nn_buttons']


    # handle image upload from Dropzone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename
            )
            # append image urls
            file_urls.append(filename)
            # file_urls.append(photos.url(filename))
        for url in file_urls:
            print(url)
            # response = requests.get(url)
            # img = Image.open(BytesIO(response.content))
            img = Image.open(open(f'./static/manager/{url}','rb'))
            img_o = img.resize((100, 100))
            img_b = img.resize((256, 256))
            img  = trans_c(img_o)
            output = model_c(img.unsqueeze(0))
            c = class_c[np.argmax(output.data.numpy())]
            nn_colours.append(c)
            img  = trans_s(img_o)
            output = model_s(img.unsqueeze(0))
            print(output.size())
            s = class_s[np.argmax(output.data.numpy())]
            nn_sleeves.append(s)
            img  = trans_b(img_b)
            output = torch.sigmoid(model_b(img.unsqueeze(0))).data[0].item()
            print(output)
            b = class_b[int(output)]
            nn_necklines.append(b)
            # nn_sleeves.append(c)
            # img  = trans_n(img_o)
            # output = model_n(img.unsqueeze(0))
            # n = class_n[np.argmax(output.data.numpy())]
            # nn_necklines.append(n)
            # nn_necklines.append(c)

        session['file_urls'] = file_urls
        session['nn_colours'] = nn_colours
        session['nn_sleeves'] = nn_sleeves
        session['nn_necklines'] = nn_necklines
        # session['nn_buttons'] = nn_buttons


        return "uploading..."
    # return dropzone template on GET request
    return render_template('manager.html')

@app.route('/manager/results')
def manager_results():
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('home'))
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    nn_colours = session['nn_colours']
    nn_sleeves = session['nn_sleeves']
    nn_necklines = session['nn_necklines']
    # nn_buttons = session['nn_buttons']
    session.pop('file_urls', None)
    session.pop('nn_results',None)
    session.pop('nn_sleeves',None)
    session.pop('nn_necklines',None)
    # session.pop('nn_buttons', None)
    return render_template('manager_results.html', file_urls=file_urls,
                           nn_colour=nn_colours, nn_necklines=nn_necklines,
                           nn_sleeves=nn_sleeves)

if __name__ == '__main__':
    app.run(debug=True)
