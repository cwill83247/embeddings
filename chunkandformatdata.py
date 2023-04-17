import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity


################################################################################
### Step 5
################################################################################

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

################################################################################
### Step 6
################################################################################

# Create a list to store the text files                                    ##### This then reads the text File 
#texts=[]

# Get all the text files in the text directory
#for file in os.listdir("text/" + domain + "/"):

    # Open the file and read the text
f = open('contenttoembed/textcopyandpastedplanning.csv') 
text = f.read()                                                                 #text is the contents of the file 

    # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.               ### ? 
    #texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(columns = ['fname', 'text'])                                     ## create columns readt for data 

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)                           ## calls the remove_newlines function 
df.to_csv('planningscraped.csv')                                              ##scraped csv had id,fname and text
df.head()    