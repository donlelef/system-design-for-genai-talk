from typing import Any

from langfuse.decorators import observe
from langfuse.openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient


class GuardrailModel(BaseModel):
    response: bool

@observe
def main():
    qdrant_client = QdrantClient("http://localhost")
    openai_client = OpenAI()

    query = "The movie talks about killing a king."
    print(f"ORIGINAL QUERY: {query}")

    if check_violence_in_text(openai_client, query):
        print("GUARDRAIL FAILED: The query contains references to violence or hate.")
    else:
        print("GUARDRAIL PASSED: The query does not contain references to violence or hate.")

    expanded_query = expand_query(openai_client, query)
    print(f"EXPANDED QUERY: {expanded_query}")

    hyde_description = create_hypothetical_movie_overview(openai_client, query)
    print(f"HYDE QUERY: {hyde_description}")

    query_embedding = embed_text(openai_client, expanded_query)
    hyde_query_embedding = embed_text(openai_client, hyde_description)
    print(f"QUERY EMBEDDING: {query_embedding}")
    print(f"HYDE EMBEDDING: {hyde_query_embedding}")

    payloads = get_payload_from_embedded_query(qdrant_client, query_embedding)
    hyde_payloads = get_payload_from_embedded_query(qdrant_client, hyde_query_embedding)
    all_payloads = payloads + hyde_payloads

    ranked_movies = rerank_movies(openai_client, all_payloads, query)
    print(f"RANKED MOVIES: \n {ranked_movies}")

    generated_answer = generate_answer(openai_client, query, ranked_movies)
    print(f"GENERATED ANSWER: {generated_answer}")


def generate_answer(openai_client:OpenAI, query:str, context: str) -> str:
    system_prompt = """
      You are a DVD record store assistant and your goal is to recommend the user with a good movie to watch.

      You are a movie expert and a real geek: you love sci-fi movies and tend to get excited when you talk about them.
      Nevertheless, no matter what, you always want to make your customers happy.
    """
    prompt = f"""
      Here are some suggested movies (ranked by relevance) to help you with your choice.
      {context}

      Use these suggestions to answer this question:
      {query}
    """
    generation_answer = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    generated_answer = generation_answer.choices[0].message.content
    return generated_answer


def rerank_movies(openai_client: OpenAI, payloads: list[dict[str, Any]], query:str) -> str:
    reranking_prompt = f"""
    We want to find the most relevant movie to the following query:
    {query}
    Here is a list of suggested movies:
    {format_records_into_context(payloads)}
    
    Remove all the movies that are note relevant to the query.
    Return a list of the most relevant movies ranked by decreasing relevance.
    Keep the original format and do not add any additional information.
    """
    reranking_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": reranking_prompt},
        ],
    )
    ranked_movies = reranking_response.choices[0].message.content
    return ranked_movies

@observe
def get_payload_from_embedded_query(
        qdrant_client: QdrantClient,
        query_embedding:list[float],
        collection_name:str="movies"
) -> list[dict[str, Any]]:
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5
    )
    payloads = [r.payload for r in search_result]
    return payloads

@observe
def embed_text(openai_client: OpenAI, text:str, embedding_model:str="text-embedding-3-small") -> list[float]:
    embedding_response = openai_client.embeddings.create(
        input=text,
        model=embedding_model
    )
    return embedding_response.data[0].embedding

@observe
def create_hypothetical_movie_overview(openai_client: OpenAI, query: str) -> str:
    hyde_prompt = f"""
    You have been given the following description of a movie: "{query}".
    Create a short description of a movie that would be relevant to the given description.
    Only answer with the movie description.
    """
    hyde_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": hyde_prompt},
        ],
    )
    hyde_description = hyde_response.choices[0].message.content
    return hyde_description

@observe
def expand_query(openai_client: OpenAI, query: str) -> str:
    query_expansion_prompt = f"""
    You have been given the following description of a movie: "{query}".
    Improve the description with more details to make it more informative.
    Only answer with the improved description.
    """
    query_expansion_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": query_expansion_prompt},
        ],
    )
    expanded_query = query_expansion_response.choices[0].message.content
    return expanded_query

@observe
def check_violence_in_text(openai_client:OpenAI, query:str) -> bool:
    guardrail_prompt = f"""
    Check the text after the --- to ensure that it does not contain any reference to violence or hate:
    ---
    {query}
    Return a boolean value indicating whether the text contains any reference to violence or hate.
    """
    guardrail_response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": guardrail_prompt},
        ],
        response_format=GuardrailModel,
    )
    return guardrail_response.choices[0].message.parsed.response

@observe
def format_records_into_context(payloads: list[dict[str, Any]]) -> str:
    context_template = """
    Title: {title}
    Overview: {overview}
    Release date: {release_date}
    Runtime: {runtime}
    """
    formatted_template = "".join(
        context_template.format(
            title=p["title"],
            overview=p["overview"],
            release_date=p["release_date"],
            runtime=p["runtime"],
        )
        for p in payloads
    )
    return formatted_template


if __name__ == '__main__':
    main()