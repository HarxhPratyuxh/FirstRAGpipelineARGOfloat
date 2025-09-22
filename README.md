# Argo Float AI RAG Chatbot
# 📝 Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to create an intelligent chatbot. The chatbot is designed to answer questions and explain concepts based on the contents of a specific document about the Argo Float project. It uses Google's Gemini models for both embedding the document text and generating answers, ensuring high-quality, context-aware responses.
# ✨ Features
-Document Loading: Loads text directly from a .docx file.
-Text Chunking: Splits the document into smaller, manageable chunks for efficient processing.
Vector Embeddings: Utilizes Google's embedding-001 model to create numerical representations (vectors) of the text chunks.
Vector Storage: Stores and indexes the vectors using ChromaDB for fast and efficient retrieval.
RAG Chain: Implements a complete RAG pipeline using LangChain.
Intelligent Q&A: Leverages the gemini-2.5-flash-lite model to understand questions and generate comprehensive answers based on the retrieved document context.
Secure API Key Handling: Uses a .env file to manage the Google API key securely.
💻 Tech Stack
Language: Python 3.x
Core Libraries:
langchain: For building the RAG pipeline and orchestrating the components.
langchain-google-genai: To integrate Google's Gemini models.
langchain-community: For community-supported integrations like document loaders.
chromadb: For creating and managing the local vector store.
docx2txt: For extracting text from .docx files.
python-dotenv: For managing environment variables.
🚀 Getting Started
Follow these steps to set up and run the project on your local machine.
1. Prerequisites
Python 3.8 or higher
A Google API key with access to the Gemini API. You can obtain one from Google AI Studio.
2. Clone the Repository
Clone this repository to your local machine:
git clone <your-repository-url>
cd <your-repository-directory>


3. Set Up a Virtual Environment
It's recommended to use a virtual environment to manage project dependencies.
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


4. Install Dependencies
Install all the required Python packages using the requirements.txt file.
pip install -r requirements.txt


5. Configure Environment Variables
Create a file named .env in the root directory of the project.
Add your Google API key to this file as shown below:
GOOGLE_API_KEY="AIzaSy...Your...Secret...API...Key"


6. Add the Source Document
Place your source document (FloatChat- Team Argonauts .docx) in the root directory of the project.
If your document has a different name, make sure to update the filename in the pratyush_first_rag_pipeline.py script.
▶️ How to Run
Once the setup is complete, you can run the RAG pipeline with a single command:
python pratyush_first_rag_pipeline.py


The script will then:
Load the document.
Split it into chunks.
Create and store vector embeddings in a local ./chroma_db directory.
Execute the predefined questions and print the generated answers to the console.
You can modify the questions inside the script by changing the strings passed to the rag_chain.invoke() function at the end of the file.
🔧 How It Works
The project follows the standard RAG architecture:
Loading: The Docx2txtLoader reads the content from the specified Word document.
Splitting: The RecursiveCharacterTextSplitter breaks the document into smaller text chunks, which is crucial for the embedding model's context limits.
Embedding & Storing: Each chunk is converted into a vector embedding by the Gemini model. These embeddings are stored in a Chroma vector database. This process is done only once and the database is persisted locally.
Retrieval: When a user asks a question, the question is also embedded. The system then queries the vector database to find the text chunks with embeddings that are most similar to the question's embedding.
Generation: The original question along with the retrieved, relevant text chunks are passed to the Gemini LLM. The model then generates a human-like, comprehensive answer based on the provided context.
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
