# Document Agent: Upload + Q&A + Summarize App
# Streamlit + LangChain + Gemini

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_experimental.tools import PythonREPLTool
import tempfile
import os
import getpass
import google.generativeai as genai     # Optional: for explicit API key configuration if not using env var

# --- IMPORTANT: Configure your Google API Key ---
# Option 1: Set as an environment variable (recommended)
# Ensure your GOOGLE_API_KEY environment variable is set.
# For example, in your terminal: export GOOGLE_API_KEY="YOUR_API_KEY"
# LangChain will automatically pick it up.

# Option 2: Configure directly in the script (less recommended for production)
# If you haven't set the environment variable, you can uncomment and use the following:

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

st.set_page_config(page_title="Document Agent (Gemini)", layout="wide")
st.title("📄 Document Agent with Gemini: Ask Questions, Analyze, Summarize PDFs")

# Check if GOOGLE_API_KEY is available
if not os.getenv("GOOGLE_API_KEY") and not (locals().get("GOOGLE_API_KEY") and genai.conf.api_key):
    st.error("🚨 Google API Key not found. Please set the GOOGLE_API_KEY environment variable or configure it in the script.")
    st.stop()


uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    try:
        st.info("Processing PDF... ⏳")
        # Load and split document
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # Adjusted chunk size for potentially different tokenization
        chunks = splitter.split_documents(pages)

        if not chunks:
            st.error("Could not extract text from the PDF. The document might be empty, image-based without OCR, or corrupted.")
            st.stop()

        # Embedding + Vector Store
        st.info("Generating embeddings and creating vector store... ✨")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Define LLM
        # Make sure to use a model that supports agent usage well, e.g., gemini-1.5-pro-latest
        # For older models like 'gemini-pro', you might need `convert_system_message_to_human=True`
        # depending on the agent type and how it handles system prompts.
        llm = ChatGoogleGenerativeAI(
            #model="gemini-1.5-pro-latest", # or "gemini-pro"
            model="gemini-2.0-flash",
            temperature=0.3,
            # convert_system_message_to_human=True # May be needed for older models or specific agent types
        )

        # Define tools
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        rag_tool = Tool(
            name="PDF_QA_Tool", # Tool names should not have spaces
            func=rag_chain.run, # Use .run for the func parameter
            description="Useful for answering questions about the content of the uploaded PDF document. Input should be a clear question."
        )

        python_repl_tool = PythonREPLTool()
        # For safety, you might want to customize or sandbox the PythonREPLTool in a real application

        tools = [rag_tool, python_repl_tool]

        # Initialize agent
        # Note: Agent behavior can vary between models (OpenAI vs Gemini).
        # You might need to experiment with different agent types or prompt engineering for optimal results.
        # ZERO_SHOT_REACT_DESCRIPTION is a good starting point.
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True # Helpful for debugging agent issues
        )

        st.success("✅ PDF processed! Ready to answer your questions.")
        user_input = st.text_input("Your question or task:", placeholder="e.g., Summarize the key findings, or ask a specific question.")

        if user_input:
            with st.spinner("🤖 Gemini is thinking..."):
                try:
                    response = agent.run(user_input)
                    st.markdown("### 💡 Response:")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred while processing your request: {e}")
                    st.error("This might be due to the agent's decision-making process, API limitations, or the nature of the query. Try rephrasing your question or breaking it into smaller parts.")

    except Exception as e:
        st.error(f"An error occurred during PDF processing or agent initialization: {e}")
    finally:
        # Clean up temp file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
else:
    st.info("👋 Please upload a PDF document to begin.")
