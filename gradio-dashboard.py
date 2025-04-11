import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

# give a better resolution book cover back from google books
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)


raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

# Define where to store your persistent vector database
persist_path = "chroma_books_db"
embedding = OpenAIEmbeddings()

if os.path.exists(persist_path):
    # ✅ Load existing persistent DB
    db_books = Chroma(
        persist_directory=persist_path,
        embedding_function=embedding
    )
else:
    # ✅ Create new DB and persist
    db_books = Chroma.from_documents(
        documents,
        embedding=embedding,
        persist_directory=persist_path
    )

def retrive_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:
    
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)
    
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs[:final_top_k]

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Fear":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Surprise":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True) 
    elif tone == "Disgust":
        book_recs.sort_values(by="disgust", ascending=False, inplace=True)
    
    return book_recs

def recommend_books(
    query: str,
    category: str,
    tone: str
):
    recommendations = retrive_semantic_recommendations(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
            
        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommendations")
    
    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g. I want to read a book about a detective solving a mystery in a small town."
                                )
        category_dropdown = gr.Dropdown(label = "Select a category:", choices = categories, value = "All")
        tone_dropdown = gr.Dropdown(label = "Select a tone:", choices = tones, value = "All")
        submit_button = gr.Button("Get Recommendations")
        
        
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns = 8, rows = 2)
    
    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)
    
    
if __name__ == "__main__":
    dashboard.launch()
        
        
    