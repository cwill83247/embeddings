print ("hi")

import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding        # this makes the mebeddings easier and is part of open ai's API
from openai.embeddings_utils import cosine_similarity      # need this to caulcaulate the simailirties between the vectors 

#open.api_key = os.environ

df = pd.read_csv('contenttoembed/demowords.csv')   # pandadata frame 
print(df)

# is creating a new dataframe column called "embedding"
# apply an openai embedding process to the column "Text"  and put this value into the new column we created
#  the x is the value being grabbed from the column "text" 
#  then converts to a .csv file 
df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df.to_csv('word_embeddings.csv')


## convert the embeddigns file into DataFrames
df = pd.read_csv('word_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)
df


# prompting input
search_term = input('Enter a search term: ')

# creating a vector for the search entered above  this is using OpenAI text-embedding-adaa-002 engine""
# using the "search_term"
search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")
search_term_vector
print(search_term_vector)



# adding a new column to DataFrame called simalatrities
# calculating the simalirty of the search term vs the text "row"
# put the cosine simailirty into it for each "text" row (so X)  and comparing to "search_term_vector"

df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
df
# sort the top 20 based on simalairity score 
# slight sytax change as wasn;t working form tutorial 
dfsorted = df.sort_values(by="similarities", ascending=False).head(20)
print('here in code')
print(dfsorted)