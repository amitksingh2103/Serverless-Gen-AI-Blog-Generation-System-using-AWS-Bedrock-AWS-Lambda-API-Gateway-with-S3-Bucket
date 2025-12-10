import json, os, base64
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        query = body.get("query")
        pdf_file_b64 = body.get("pdf_file")

        if not query:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing query"})}

        if not pdf_file_b64:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing PDF file"})}

        # Decode PDF
        pdf_bytes = base64.b64decode(pdf_file_b64)
        pdf_path = "/tmp/temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Load PDF and split into chunks
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)

        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        index_name = "rag-pdf-index"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        index = pc.Index(index_name)

        # Embed chunks
        embed_model = OpenAIEmbeddings()
        texts = [doc.page_content for doc in splits]

        embeddings = embed_model.embed_documents(texts)

        # Convert to Pinecone format
        vectors = [
            {"id": f"chunk_{i}", "values": embeddings[i], "metadata": {"text": texts[i]}}
            for i in range(len(texts))
        ]

        # Upsert into Pinecone
        index.upsert(vectors)

        # Query Pinecone
        embedded_query = embed_model.embed_query(query)
        result = index.query(vector=embedded_query, top_k=5, include_metadata=True)

        # Build context for RAG
        context_text = "\n\n".join([match["metadata"]["text"] for match in result["matches"]])

        # LLM (Groq)
        groq_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_key)

        system_prompt = """Use ONLY the provided context to answer the user.
If the answer is not in the context, say: 'I don't know'.

Context:
{context}
"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        rag_chain = create_stuff_documents_chain(llm, qa_prompt)

        response = rag_chain.invoke({"input": query, "context": context_text})

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"answer": response["answer"]})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }