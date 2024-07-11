from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.tools import Tool
from .tools_constants import retriever_tool_description, few_shots_examples

def get_retriever_tool():
    embeddings = OpenAIEmbeddings()

    few_shot_docs = [
        Document(
            page_content=question, metadata={"sql_query": few_shots_examples[question]}
        )
        for question in few_shots_examples.keys()
    ]
    vector_db = FAISS.from_documents(few_shot_docs, embeddings)
    retriever = vector_db.as_retriever()

    def retriever_func(query: str) -> str:
        results = retriever.get_relevant_documents(query)
        return results[0].page_content if results else "No relevant examples found."

    return Tool(
        name="sql_get_few_shot",
        description=retriever_tool_description,
        func=retriever_func
    )