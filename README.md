# RE-SEARCH

DEMO FOLDER:
contains the web application used during demo/presentation. 

localhost:5000   ---> HOME PAGE
/customer ---> CUSTOMER SIDE
/manager ---> MANAGER SIDE

Customer side search engine which queries a **LOCAL** db to obtain search results and return images corresponding to key word search.

Manager side allows upload of images that get preprocessed and ran through saved models to return predicted labels.

DEMO/CUSTOMER.PY:

calls customer_helpers.py with inputted search keywords to obtain the paths of the image results.


DEMO/CUSTOMER_HELPERS.PY:

stores functions called by customer.py that generates query based on inputted search keywords and returns the paths of the image results.


MAIN.PY:

trains a NN model based on specified parameters and dataset specified through datafolder path. Will show plots, confusion matrices and save model. 

MODEL.PY:

contains models called on by main file and the architecture of final sleeves, necklines, buttons and colour models


PREPROCESS.PY:

preprocesses all images scraped from web stored in webscraping/data folder, resizes and accumulates images into new test folders for main.py


INTEGRATE.PY:

runs all images stored in the input folder location through the model stored and outputs the label and confidence of the images in a list.


PROCESS.PY:

for images in a specified folder, calls integrate.py to generate predictions and labels for each clothing feature type and stores the results in a local database in a standardized form.


TEST.PY:

stores functions that generate queries to store and search through a local database.


MODELS FOLDER:

contains all the saved best models so far...

WEBSCRAPING FOLDER

contains scripts for scraping stores
