import pandas as pd
import tqdm
from openai import OpenAI
from qdrant_client import models, QdrantClient


def main():
    movies_df = fetch_movies()

    qdrant = QdrantClient("http://localhost")
    collection_name = "movies"
    delete_and_create_collection(qdrant, collection_name)

    openai_client = OpenAI()
    embedding_model = "text-embedding-3-small"
    embedding_response = openai_client.embeddings.create(
        input=movies_df["overview"].tolist(), model=embedding_model
    )
    movies_df["embedding"] = [e.embedding for e in embedding_response.data]

    records = transform_movie_dataframe_to_records(movies_df)
    qdrant.upload_points(collection_name=collection_name, points=records)


def transform_movie_dataframe_to_records(
    movies_df: pd.DataFrame,
) -> list[models.Record]:
    return [
        models.Record(
            id=idx,
            vector=mov["embedding"],
            payload={
                "title": mov["title"],
                "overview": mov["overview"],
                "release_date": mov["release_date"],
                "runtime": mov["runtime"],
                "genre": mov["genre"],
            },
        )
        for idx, mov in tqdm.tqdm(movies_df.iterrows())
    ]


def delete_and_create_collection(
    qdrant: QdrantClient, collection_name: str, embedding_dimensions: int = 1536
):
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dimensions, distance=models.Distance.COSINE
        ),
    )


def fetch_movies() -> pd.DataFrame:
    movies_df = pd.read_parquet(
        "https://github.com/xtreamsrl/genai-for-engineers-class/raw/main/data/movies.parquet"
    )
    movies_df = movies_df.iloc[:300, :]
    return movies_df


if __name__ == "__main__":
    main()
