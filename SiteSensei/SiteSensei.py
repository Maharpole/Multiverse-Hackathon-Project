import os
import uuid
import pickle
import faiss
import httpx
import asyncio
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from fake_useragent import UserAgent
from flask import Flask, jsonify, request
from flask import render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from playwright.async_api import async_playwright
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.document_loaders import S3FileLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.docstore.document import Document
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from urllib.parse import urlparse, urljoin
from unstructured.partition.html import partition_html
import mysql.connector
import boto3
import sqlite3


app = Flask(__name__)
limiter = Limiter(key_func=get_remote_address)
processed_urls = set()
DATABASE = 'sitesensei.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # This allows for dictionary-like access to row results
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

#URL Loader

async def load_urls(xmlurls):

    urls = xmlurls

    loader =  UnstructuredURLLoader(urls=urls, remove_selectors=["header", "footer"])

    data = loader.load()
    
    return data

#Connect to Database

def connect_db():
    connection = sqlite3.connect('sitesensei.db')  # The database file
    return connection

connection = connect_db()

#Root Route

@app.route('/')
def index():
    return render_template('index.html')

#Create User Endpoint If 

#Process Sitemap Endpoint

@limiter.limit("3 per minute")
@app.route('/processSitemap', methods=['POST'])
async def process_sitemap():
    
    sitemap_url = request.form['sitemap_url']
    user_id = request.args.get('user_id', '')
    
    # Check if the URL is a valid domain
    
    parsed_url = urlparse(sitemap_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        return jsonify(status="Invalid domain", response=response.status_code)

    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.head(sitemap_url)
            if response.status_code != 200:
                return jsonify(status="Sitemap URL not accessible", response=response.status_code)
    except httpx.RequestError as e:
        print(f"Error accessing {sitemap_url}: {e}")
        return jsonify(status="Error accessing URL", error=str(e))

    xmlurls = await scrape_sitemap(sitemap_url)
    
    # Scrapes pages in "urls".
    
    remove_selectors = ["header", "footer"]  # Optional: List of selectors to remove
    
    data = await load_urls(xmlurls)
    
    
    # Text Splitter

    text_splitter = CharacterTextSplitter(
                                        separator='\n', 
                                        chunk_size=1000, 
                                        chunk_overlap=200,
                                        length_function = len,
                                        )

    docs = text_splitter.split_documents(data)
    
    #Ai Embedding

    embeddings = OpenAIEmbeddings(deployment='text-embedding-ada-002',
                model='text-embedding-ada-002',
                openai_api_version='2020-11-07',
                openai_api_base=None,
                openai_api_type=None,
                embedding_ctx_length=8191,
                openai_api_key=None,
                openai_organization=None,
                allowed_special=set(),
                disallowed_special='all',
                chunk_size=1000,
                max_retries=6)

    vectorStore_openAI = FAISS.from_documents(docs, embeddings)

    #generate a unique API key
    api_key = str(uuid.uuid4())
    s3 = boto3.client('s3')
    bucket_name = 'sitesenseiembedfiles'
    object_name = f"SiteSensei_Embed_Files/faiss_store_openai_{api_key}.pkl"

     # Serialize the object
    serialized_vector_store = vectorStore_openAI.serialize_to_bytes()

    # Upload the serialized object to S3
    s3.put_object(Bucket=bucket_name, Key=object_name, Body=serialized_vector_store)
    
    # Insert the API key and file path into the database
    insert_api_key_and_file_path(user_id, api_key, object_name)

    return jsonify(
        status="Sitemap processed", 
        api_key=api_key
        )

async def scrape_sitemap(url):
    if url in processed_urls:  # Check if the URL has already been processed
        return []
    processed_urls.add(url)  # Mark the URL as processed

    urls = []
    xml_urls = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        all_urls = await page.eval_on_selector_all('a', 'elements => elements.map(element => element.href)')

        # Extract the domain of the sitemap URL
        sitemap_domain = urlparse(url).netloc

        # Keep only URLs that have the same domain as the sitemap URL
        for u in all_urls:
            if urlparse(u).netloc == sitemap_domain:
                if u.endswith('.xml'):
                    # If the URL ends with .xml, recursively scrape it
                    xml_urls.extend(await scrape_sitemap(u))
                else:
                    urls.append(u)

        await browser.close()

    # Combine the URLs from the current page with the URLs from any .xml pages
    urls.extend(xml_urls)
    
    return urls

# function to insert api key and file path into their respective tables

def insert_api_key_and_file_path(user_id, api_key, object_name):
    cursor = connection.cursor()
    query = "INSERT INTO Key_Associations (user_id, api_key, s3_object_key) VALUES (%s, %s, %s)"
    values = (user_id, api_key, object_name)
    cursor.execute(query, values)
    connection.commit()
    cursor.close()

# Ask Question Endpoint

@limiter.limit("10 per minute")
@app.route('/askQuestion', methods=['POST'])
def ask_question():
    from langchain import PromptTemplate
    from langchain import LLMChain

    embeddings = OpenAIEmbeddings(deployment='text-embedding-ada-002',
            model='text-embedding-ada-002',
            openai_api_version='2020-11-07',
            openai_api_base=None,
            openai_api_type=None,
            embedding_ctx_length=8191,
            openai_api_key=None,
            openai_organization=None,
            allowed_special=set(),
            disallowed_special='all',
            chunk_size=1000,
            max_retries=6)

    #retrieve active user ID, API key, and the question.
    
    user_id = request.json['user_id']
    api_key = request.json['api_key']
    question = request.json['question']
    
    #look up associated s3 object key from given api and user key
    
    s3_object_key = get_key_from_db(user_id, api_key)

    s3 = boto3.client('s3')
    bucket_name = 'sitesenseiembedfiles'
    object_name = f"SiteSensei_Embed_Files/faiss_store_openai_{api_key}.pkl"

    # Download the object from S3
    s3_object = s3.get_object(Bucket=bucket_name, Key=object_name)
    

    bytes = s3_object['Body'].read()


    ####bytes = pickle.loads(object_content)
    
    # Deserialize the object
    vectorstore = FAISS.deserialize_from_bytes(serialized=bytes, embeddings=embeddings)
    
    llm=ChatOpenAI(temperature=0.9, model_name='gpt-3.5-turbo-0301')
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    Facts = chain({"question": question}, return_only_outputs=True)
    
    prompt = PromptTemplate(
    input_variables=["question", "Facts"],
    template= "please answer the following question as a support representative would and include the link to the source if applicable: {question}? Here is the information to use to answer the question: {Facts}",
    )
    personality_chain = LLMChain(llm=llm, prompt=prompt)

    response = personality_chain.run({
            'question': question, 
            'Facts': Facts
            })

    # Create a JSON response to send back to the client
    response_json = { 
        "status": response
    }

    return jsonify(response_json)
    
    # Retrieve the selected key from inputs
    
def get_key_from_db(user_id, api_key):
    cursor = connection.cursor()
    query = "SELECT s3_object_key FROM Key_Associations WHERE user_id = %s AND api_key = %s"
    values = (user_id, api_key)
    cursor.execute(query, values)
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else None

    return jsonify(response_json)

if __name__ == '__main__':
    app.run(debug=False, port=8000, host='0.0.0.0')


