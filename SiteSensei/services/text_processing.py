from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.text_splitters import CharacterTextSplitter

def load_and_split_text(urls):
    loaders = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
    data = loaders.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(data)
