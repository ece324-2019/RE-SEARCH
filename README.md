# RE-SEARCH

DEMO FOLDER:
contains the web application used during demo/presentation. 

localhost:5000   ---> HOME PAGE
/customer ---> CUSTOMER SIDE
/manager ---> MANAGER SIDE

Customer side search engine which queries a **LOCAL** db to obtain search results and return images corresponding to key word search.

Manager side allows upload of images that get preprocessed and ran through saved models to return predicted labels.

MAIN.PY:

trains a NN model based on specified parameters and dataset specified through datafolder path. Will show plots, confusion matrices and save model. 

MODEL.PY:

contains models called on by main file and the architecture of final sleeves, necklines, buttons and colour models


PREPROCESS.PY:

preprocesses all images scraped from web stored in webscraping/data folder, resizes and accumulates images into new test folders for main.py

GRID_SEARCH.PY:

Script used to run a hyperparameter grid search to optimize model.

MODELS FOLDER:

contains all the saved best models so far...

WEBSCRAPING FOLDER

contains scripts for scraping stores
