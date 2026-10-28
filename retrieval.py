from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Load DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ✅ FIXED LLM
llm = Ollama(model="llava")

# Prompt
prompt_template = """
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# Chat loop
while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    result = qa.invoke({"query": query})   # ✅ updated
    print("\nAnswer:\n", result["result"])