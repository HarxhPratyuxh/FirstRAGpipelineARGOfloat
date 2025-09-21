import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma # Corrected import for the new version
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Setup ---
# Load environment variables from a .env file for better security
load_dotenv()

# Get your Google API key from environment variables
# IMPORTANT: Create a file named '.env' in the same directory and add your key like this:
# GOOGLE_API_KEY="AIzaSy..."
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable in a .env file.")

# --- 1. Load Document ---
print("📄 Loading document...")
try:
    # Make sure this is the correct path to your file
    loader = Docx2txtLoader("FloatChat- Team Argonauts .docx")
    docs = loader.load()
    print(f"✅ Document loaded successfully. It has {len(docs[0].page_content)} characters.")
except Exception as e:
    print(f"❌ Error loading document: {e}")
    exit()

# --- 2. Split Document into Chunks ---
print("쪼 Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"✅ Document split into {len(splits)} chunks.")

# --- 3. Create Vector Store ---
print("🧠 Creating vector store with Gemini embeddings...")
# Initialize Google Generative AI Embeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

# Create the vectorstore using Gemini embeddings
# This will save the embeddings to a local directory for reuse
vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings, persist_directory="./chroma_db")
print(f"✅ Vector store created. It contains {vectorstore._collection.count()} embeddings.")

# --- 4. Create RAG Chain ---
print("🔗 Building RAG chain...")
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for explaining the argo float AI rag chatbot project, by question-answering and explaining the project aspects in details. Use the following pieces of retrieved context to answer the question and explain the project details and working. If you don't know the answer, just say that you don't know. keep the answer comprehensive and detailed with topic division or bullet points."),
    ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:")
])

# Using a standard, publicly available and powerful model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("✅ RAG chain built successfully.")


# --- 5. Invoke Chain and Print Results ---
print("\n🚀 Running queries...")

# Query 1
print("\n❓ Question 1: What is the tech stack of the argo float project?")
try:
    response1 = rag_chain.invoke("tech stack of argo float project")
    print("\n💡 Answer 1:")
    print(response1)
except Exception as e:
    print(f"❌ Error during invocation: {e}")


# Query 2
print("\n" + "="*50)
print("\n❓ Question 2: Explain the argo float lifecycle and how and what data is collected.")
try:
    response2 = rag_chain.invoke("explain the argo float lifecycle and how and what data is collected")
    print("\n💡 Answer 2:")
    print(response2)
except Exception as e:
    print(f"❌ Error during invocation: {e}")
    

# Query 3
print("\n❓ Enter custom query")
try:
    response3 = rag_chain.invoke(input("enter your query"))
    print("\n💡 Answer 3:")
    print(response3)
except Exception as e:
    print(f"❌ Error during invocation: {e}")

print("\n\n✅ All queries processed.")
