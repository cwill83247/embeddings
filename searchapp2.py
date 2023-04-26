from flask import Flask, request, render_template
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import config

app = Flask(__name__)

#openai.api_key = config.OPENAI_API_KEY
openai.api_endpoint = "v1/chat/completions"

@app.route('/static/<path:filename>')
def serve_static(filename):
  return app.send_static_file(filename)

@app.route('/')
def search_form():
  return render_template('search_form.html')


@app.route('/search')
def search():
    # Get the search query from the URL query string
    query = request.args.get('query')

    search_term_vector = get_embedding(query, engine="text-embedding-ada-002")                     # getting the vector for the search term
    print(search_term_vector)  

    #df = pd.read_csv('reviews_with_embeddings_1k.csv')                                          # putting csv in to dataframe likely to use pinecone db
    df = pd.read_csv('planningwithembeddings.csv') 
    print (df)      

    #df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    print('after adding the simalaraties column in the df')
    print (df)

    #df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    df["similarities"] = df['embeddings'].apply(lambda x: cosine_similarity(x, search_term_vector))
    sorted_by_similarity = df.sort_values("similarities", ascending=False).head(3)               # limiting results to top 3 based on simalarities 

    results = sorted_by_similarity['text'].values.tolist()    # what column is being output to results 
    
## How do I take the vector type info and then generate conversational element ?

    # 26/4/23 added
    if len(results) == 0:
        return render_template('search_results.html', query=query, message="Sorry, no results for your search. Please try again.")


   # 26/4/23  Use GPT-3 to generate conversational responses to the search results
    #openai.api_key = config.OPENAI_API_KEY
    #model_engine = "text-davinci-002"
    model_engine = "text-davinci-002"
    prompt = f"Here are the top three results that match your search for '{query}':\n"
    for i, result in enumerate(results):
        prompt += f"{i+1}. {result}\n"
    prompt += "What would you like to know more about?\n"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        #max_tokens=1024,
        max_tokens=2024,
        n=1,
        stop=None,
        temperature=0.5,
        ### had to declare as a variable api_endpoint="https://api.openai.com/v1/chat/completions",    # added
    )  

    print("repsonse below")
    print(response)  
    print("i outpout")
    print(i)
    print("result below")
    print (result)
    # 26/4/23 --  ????
    #conversation_response = response.choices[0].text.strip()
    conversation_response = response.choices

    # Render the search results template, passing in the search query and results
    #return render_template('search_results.html', query=query, results=results)
    #26/4/23
    return render_template('search_results.html', query=query, results=conversation_response)   # returning back to template to output in HTML

if __name__ == '__main__':
  app.run()