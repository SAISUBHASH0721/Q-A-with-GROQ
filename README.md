This project is a Streamlit-based Q&A application that leverages Retrieval-Augmented Generation (RAG) using the ChatGroq model with Llama3-8b-8192 to answer questions based on the content of research papers. The application creates a vector database using FAISS from uploaded PDF documents and enables users to query this database for relevant information.

Key Features
Document Upload and Embedding Creation:

Users can upload PDF documents to the application.
The application uses PyPDFDirectoryLoader to load documents from the specified directory.
Documents are split into manageable chunks using RecursiveCharacterTextSplitter.
FAISS (Facebook AI Similarity Search) is utilized to create vector embeddings from the processed document chunks, allowing for efficient and scalable similarity searches.
Retrieval-Augmented Generation (RAG):

RAG combines retrieval and generation techniques to provide answers based on retrieved documents. This ensures that the generated response is grounded in factual data from the provided documents.
The ChatGroq model with Llama3-8b-8192 is used to generate responses to user queries, guided by a ChatPromptTemplate that formats the input context and questions.
Question Answering Interface:

Users can input questions related to the uploaded research papers.
The application retrieves relevant document chunks using FAISS and feeds them into the LLM to generate an accurate response.
The response is displayed in the Streamlit interface, providing a clear and concise answer.
Document Similarity Search:

An expander widget in Streamlit displays similar documents or document chunks retrieved based on the user's query.
This feature allows users to understand the context and source of the information provided in the answer, enhancing transparency and reliability.
Technology Stack
Frontend: Streamlit for building an interactive user interface.
Backend:
LangChain for managing LLM workflows and creating retrieval chains.
FAISS for efficient vector similarity search and retrieval.
GROQ API and OpenAI API for language model operations and embedding creation.
Models:
ChatGroq using the Llama3-8b-8192 model for generating natural language answers.
OpenAIEmbeddings for creating vector embeddings from document text.
How It Works
Load Environment Variables:

The application loads the required API keys from a .env file to authenticate with OpenAI and GROQ APIs.
Create Vector Embeddings:

When the "Document Embedding" button is clicked, the application loads PDF files from the Papers directory and splits them into chunks.
Embeddings for these chunks are created using the OpenAIEmbeddings model and stored in a FAISS vector database.
Generate Answers:

Users can enter questions in a text input field. When a question is submitted, the application retrieves the most relevant document chunks using FAISS.
The retrieved chunks are passed to the ChatGroq model, which generates a response based on the provided context.
Display Results:

The generated answer is displayed in the main interface.
An expandable section shows the document chunks that were most similar to the user query, providing context and source validation.
