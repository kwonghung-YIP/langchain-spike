import datetime, json
from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisVectorStore
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_core.documents import Document
from redis import Redis
from bs4 import BeautifulSoup

def parseHTMLContent(html:str) -> str:
    soup=BeautifulSoup(html)
    return soup.get_text()

embeddings=OllamaEmbeddings(model="llama3.2")

redisClient=Redis.from_url("redis://localhost:6379")

# Reference for RedisVectorStore from langchain
# https://python.langchain.com/docs/integrations/vectorstores/redis/
# https://python.langchain.com/api_reference/redis/vectorstores/langchain_redis.vectorstores.RedisVectorStore.html
# https://api.python.langchain.com/en/latest/community/vectorstores/langchain_community.vectorstores.redis.base.Redis.html

vectorStore=RedisVectorStore(
    index_name='jobs',
    embeddings=embeddings,
    redis_client=redisClient
)

loader=DirectoryLoader(
    "reed.co.uk/job-details",
    glob="*.json",
    sample_size=5,
    randomize_sample=True,
    show_progress=True,
    loader_cls=JSONLoader,
    loader_kwargs={
        'jq_schema':'.',
        'text_content':False,
    },
)
jsonFiles=loader.load()

for i in range(len(jsonFiles)):
    job=json.loads(jsonFiles[i].page_content)
    metadata={
        'employerId':job['employerId'],
        'jobId':job['jobId'],
        'jobUrl':job['jobUrl'],
        #'externalUrl':job['externalUrl'],
        **jsonFiles[i].metadata
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
    doc=Document(page_content=pageContent,metadata=metadata)
    print(f"[{datetime.datetime.now()}] {i+1}. add document {jsonFiles[i].metadata['source']} into Redis vector store...")
    #print(doc.id)
    #print(doc.page_content)
    #print(doc.metadata)
    ids=vectorStore.add_documents([doc])
    #ids=vectorStore.add_texts([pageContent],metadatas=[metadata],ids=[job['jobId']])
    print(ids[0])

