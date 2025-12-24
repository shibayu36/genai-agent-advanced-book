from langchain.tools import tool
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.configs import Settings

# 検索結果の最大取得数
MAX_SEARCH_RESULTS = 3


class SearchQueryInput(BaseModel):
    query: str = Field(description="検索クエリ")


@tool(args_schema=SearchQueryInput)
def search_xyz_qa(query: str) -> list[dict]:
    """
    XYZシステムの過去の質問回答ペアを検索する関数。
    """

    print(f"  [Tool] search_xyz_qa: {query}")

    qdrant_client = QdrantClient("http://localhost:6333")

    settings = Settings()
    openai_client = OpenAI(api_key=settings.openai_api_key)

    # クエリをベクトル化
    query_vector = openai_client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding

    # ベクトル検索を実行
    search_results = qdrant_client.query_points(
        collection_name="documents", query=query_vector, limit=MAX_SEARCH_RESULTS
    ).points

    outputs = []
    for point in search_results:
        if point.payload is None:
            continue
        outputs.append({"file_name": point.payload["file_name"], "content": point.payload["content"]})

    return outputs
