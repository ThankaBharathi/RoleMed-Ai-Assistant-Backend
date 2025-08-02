import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECIONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

pc=Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index(PINECONE_INDEX_NAME)

embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm=ChatGroq(temperature=0.3,model_name="llama3-8b-8192",groq_api_key=GROQ_API_KEY)


prompt=PromptTemplate.from_template("""
You are a helpful healthcare assistant.Answer the following quetion
 based only on the provided context.
                                    
    Question:{question}
                                    
    Context:{context}
                                    
 Include the document source if relevant in yout answer.

""")

rag_chain=prompt | llm


async def answer_query(query:str,user_role:str):

    embedding=await asyncio.to_thread(embed_model.embed_query,query)
    results=await asyncio.to_thread(index.query, vector=embedding,top_k=3,include_metadata=True)

    filtered_contexts=[]
    sources=set()

    for match in results["matches"]:
        metadata=match["metadata"]
        if metadata.get("role")== user_role:
            filtered_contexts.append(metadata.get("text","")+"\\n")
            sources.add(metadata.get("source"))


    if not filtered_contexts:
        return {"answer":"No relevant info found"}
    
    docs_text="\\n".join(filtered_contexts)
    final_answer=await asyncio.to_thread(rag_chain.invoke,{"question":query,"context":docs_text})


    return {
        "answer":final_answer.content,
        "sources":list(sources)
    }