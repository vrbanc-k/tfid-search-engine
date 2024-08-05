from contextlib import asynccontextmanager
from dataset.newsgroups20_dataset import NewsGroups20Dataset
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, constr
from search_engine.tfidf_search_engine import TfidfSearchEngine

# Document model
class Document(BaseModel):
    content: constr(min_length=1)
    name: constr(min_length=1)

global_variables = {}

# This function will run before the app starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    # (Down)Load the dataset
    newsgroup_dataset = NewsGroups20Dataset()

    # Initialize the search engine and index the dataset
    global_variables['search_engine'] = TfidfSearchEngine()
    global_variables['search_engine'].index(data=newsgroup_dataset.dataset)
    yield

    global_variables.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/search")
async def search(query: str = '', skip: int = 0, limit: int = 10) -> list[str]:
    """Performs a search using the provided search query"""

    # Perform the search using the search engine
    search_results = global_variables['search_engine'].search(search_query=query, num_of_results=limit, num_of_results_to_skip=skip)
    return search_results


@app.post("/documents")
async def index_document(document: Document) -> Document:
    """Indexes a new document into the search engine"""
    # Lowercase the document name
    name_to_be_indexed = document.name.lower()
    # Check if the document with the same name is already indexed
    if global_variables['search_engine'].check_if_document_is_indexed(name_to_be_indexed):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f'The document with the name {document.name} is already indexed')
    # Index the new document
    global_variables['search_engine'].index(zip([document.content], [name_to_be_indexed]))

    return document



    
        



