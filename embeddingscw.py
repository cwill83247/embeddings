import pandas as pd
import tiktoken
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding        # this makes the mebeddings easier and is part of open ai's API
from openai.embeddings_utils import cosine_similarity  
#from openai.embeddings_utils import get_embedding    is this need as function is in same file ?

# embedding model parameters   !!!! Can use PARAMETERS 
#embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


# Here we are loading in the CSV File    - USING TEST DATA from KAGGLE (amazon related)
# load & inspect dataset

#def get_embedding(text, model="text-embedding-ada-002"):
#   text = text.replace("\n", " ")
#   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


input_datapath = "contenttoembed/reviews_1k.csv"  # PATH to Data
df = pd.read_csv(input_datapath, index_col=0)     # Dataframe read CSV
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]    #DataFrame Fields based on CSV provided  
df = df.dropna()                                                         ## ???? Unsure
df["combined"] = (                                                            ## creating extra column called "Combined" and adding Summary and Text together
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)
df.head(2)                                                            # Output 1st 2 records and show column headings

# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

# This may take a few minutes
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))          # This is doing the embedding and adding that into column called "embedding" by calling the get_embedding function
df.to_csv("reviews_with_embeddings_1k.csv")                                                     # tjis is where the file outputs



#df = pd.read_csv('t/embedded_1k_reviews.csv')
#df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)
