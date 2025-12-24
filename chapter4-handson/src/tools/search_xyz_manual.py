from elasticsearch import Elasticsearch
from langchain.tools import tool
from pydantic import BaseModel, Field

# 検索結果の最大取得数
MAX_SEARCH_RESULTS = 3


# 入力スキーマを定義するクラス
class SearchKeywordInput(BaseModel):
    keywords: str = Field(description="全文検索用のキーワード")


# LangChainのtoolデコレーターを使って、検索機能をツール化
@tool(args_schema=SearchKeywordInput)
def search_xyz_manual(keywords: str) -> list[dict]:
    """
    XYZシステムのドキュメントを調査する関数。
    エラーコードや固有名詞が質問に含まれる場合は、この関数を使ってキーワード検索を行う。
    """

    print(f"  [Tool] search_xyz_manual: {keywords}")

    # Elasticsearchのインスタンスを作成して、ローカルのElasticsearchに接続
    es = Elasticsearch("http://localhost:9200")

    # 検索対象のインデックスを指定
    index_name = "documents"

    # 検索クエリを作成。'content' フィールドに対してキーワードで全文検索を行う
    keyword_query = {"query": {"match": {"content": keywords}}}

    # Elasticsearchに検索クエリを送信し、結果を 'response' に格納
    response = es.search(index=index_name, body=keyword_query)

    # 検索結果を格納するリスト
    outputs = []

    # 検索結果からヒットしたドキュメントを1つずつ処理
    for hit in response["hits"]["hits"][:MAX_SEARCH_RESULTS]:
        outputs.append({"file_name": hit["_source"]["file_name"], "content": hit["_source"]["content"]})

    return outputs
