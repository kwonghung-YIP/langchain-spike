import json
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from bs4 import BeautifulSoup

def parseHTMLContent(html:str) -> str:
    soup=BeautifulSoup(html)
    return soup.get_text()

def metadata_func(doc:dict,metadata:dict) -> dict:
    metadata['jobId']=doc['jobId']
    metadata['jobUrl']=doc['jobUrl']

    return metadata

loader=DirectoryLoader(
    "reed.co.uk/job-details",
    glob="*.json",
    sample_size=20,
    randomize_sample=True,
    loader_cls=JSONLoader,
    loader_kwargs={
        'jq_schema':'.',
        'text_content':False,
        'metadata_func': metadata_func,
    },
)
jsonFiles=loader.load()
print(len(jsonFiles))
print(jsonFiles[0])

docs=[]
for jsonDoc in jsonFiles:
    job=json.loads(jsonDoc.page_content)
    metadata={
        'employerId':job['employerId'],
        'jobId':job['jobId'],
        'jobUrl':job['jobUrl'],
        'externalUrl':job['externalUrl'],
    }
    pageContent=f"""
        Company: {job['employerName']}
        Location: {job['locationName']}
        Job Title: {job['jobTitle']}
        Post Date: {job['datePosted']}
        Exp Date: {job['expirationDate']}        
        Job Description:```
        {parseHTMLContent(job['jobDescription'])}
        ```
    """
    docs.append(Document(id=job['jobId'],page_content=pageContent,metadata=metadata))
    
print(len(docs))
print(docs[0])

#splitter=RecursiveJsonSplitter(max_chunk_size=300)
#print(type(docs[0].page_content))
#chunks=splitter.split_json(json.loads(docs[0].page_content))
#docs=splitter.create_documents(json.loads(docs[0].page_content))
#print(len(chunks))
#for c in chunks:
#    print(c)

# docs=splitter.create_documents(texts=[json.loads(docs[0].page_content)])
# print(len(docs))
# print(docs[0])

embeddings=OllamaEmbeddings(model="llama3.2")

vectors=embeddings.embed_documents(docs[0].page_content)
print(len(vectors))
print(docs[0],vectors[0])

vectorStore=InMemoryVectorStore(embeddings)
vectorStore.add_documents(docs)

#for index,(id,doc) in enumerate(vectorStore.store.items()):
#    print(f"{id}: {doc}")

results=vectorStore.similarity_search_with_score(query="Tech Lead in London",k=5)
print(len(results))
for doc,score in results:
    print(f"SIM:{score:3f}, {doc.metadata} {doc.page_content[:50]}")