from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient


def main():
    qdrant_client = QdrantClient("http://localhost")
    collection_name = "movies"

    openai_client = OpenAI()
    embedding_model = "text-embedding-3-small"

    query = "The movie talks about a ship."
    embedding_response = openai_client.embeddings.create(
        input=query,
        model=embedding_model
    )
    query_embedding = embedding_response.data[0].embedding

    system_prompt = """
      You are a DVD record store assistant and your goal is to recommend the user with a good movie to watch.

      You are a movie expert and a real geek: you love sci-fi movies and tend to get excited when you talk about them.
      Nevertheless, no matter what, you always want to make your customers happy.
    """

    prompt_template = """
      Here are some suggested movies (ranked by relevance) to help you with your choice.
      {context}

      Use these suggestions to answer this question:
      {question}
    """

    context_template = """
    Title: {title}
    Overview: {overview}
    Release date: {release_date}
    Runtime: {runtime}
    """

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5
    )
    payloads = [r.payload for r in search_result]

    context = format_records_into_context(payloads, context_template)
    prompt = prompt_template.format(
        context=context,
        question=query
    )

    chat_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    answer = chat_completion

    print(answer.choices[0].message.content)



def format_records_into_context(payloads: list[dict[str, Any]], template:str) -> str:
    return "".join(
        template.format(
            title=p["title"],
            overview=p["overview"],
            release_date=p["release_date"],
            runtime=p["runtime"],
        )
        for p in payloads
    )



if __name__ == '__main__':
    main()