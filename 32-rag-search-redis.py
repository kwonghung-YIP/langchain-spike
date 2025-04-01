from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisVectorStore
from redis import Redis

embeddings=OllamaEmbeddings(model="llama3.2")

redisClient=Redis.from_url("redis://localhost:6379")

# Reference for RedisVectorStore from langchain
# https://python.langchain.com/api_reference/redis/vectorstores/langchain_redis.vectorstores.RedisVectorStore.html
# https://python.langchain.com/docs/integrations/vectorstores/redis/
vectorStore=RedisVectorStore(
    index_name='jobs',
    embeddings=embeddings,
    redis_client=redisClient
)

#print(vectorStore.get_by_ids(['01JQEZT66N4XKDA9S1D9Y21F03'])[0])

query="Tech Lead role in London"

#results=vectorStore.similarity_search_with_score(query,k=5,kwargs={
#    'return_metadata':True,
#    'return_all':True,
#})
#print(len(results))
#for doc,score in results:
#    print(doc.metadata)
#    print(f"SIM:{score:3f}, {doc.metadata} {doc.page_content[:500]}")

results=vectorStore.search(query,search_type='similarity',return_all=True)
print(len(results))
for doc in results:
    print(doc.metadata)
    print(f"{doc.metadata} {doc.page_content[:500]}")


retriever=vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":1,"fetch_k":5,"lambda_mult":0.5,"return_all":True}
)
results=retriever.get_relevant_documents(query=query)
print(len(results))
print(results[0])