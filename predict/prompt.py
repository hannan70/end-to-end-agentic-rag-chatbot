from langchain.prompts import ChatPromptTemplate


# setup prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful and knowledgeable assistant. Answer the user's question using only the information provided in the <context> section below.
    <context>
    {context}
    </context>
    Question: {input}
    Instructions:
    - If the answer exists in the context, provide a clear, concise, and accurate response.
    - If the answer cannot be found in the context, reply with:
    "I'm sorry, I do not have enough information in the document to answer that."

    Only use the context provided. Do not make assumptions or use external knowledge.
    """
)