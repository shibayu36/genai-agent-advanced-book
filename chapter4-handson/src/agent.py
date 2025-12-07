"""
Step 1: 最小エージェント
- LangGraphの基本構造を理解する
- StateGraph, ノード, エッジの使い方を学ぶ
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from src.configs import Settings


# === 状態の定義 ===
# TypedDictを使って、エージェントが持つ状態を定義する
class AgentState(TypedDict):
    question: str  # ユーザーからの質問
    answer: str    # エージェントの回答

# === ノードの定義 ===
# ノードは「状態を受け取り、更新された状態を返す関数」

def create_answer(state: AgentState) -> dict:
    """質問に対する回答を生成するノード"""

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = [
        {
            "role": "system",
            "content": "あなたはXYZシステムのヘルプデスク担当です。質問に簡潔に回答してください。"
        },
        {
            "role": "user",
            "content": state["question"]
        }
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
    )

    answer = response.choices[0].message.content

    # 状態の更新部分だけを返す（全体を返す必要はない）
    return {"answer": answer}


# === グラフの構築 ===
def create_graph():
    """エージェントのグラフを構築する"""

    # StateGraphを作成（状態の型を指定）
    workflow = StateGraph(AgentState)

    # ノードを追加
    workflow.add_node("create_answer", create_answer)

    # エッジを追加
    workflow.add_edge(START, "create_answer")
    workflow.add_edge("create_answer", END) # create_answer → 終了

    # グラフをコンパイル
    app = workflow.compile()

    return app

# === グラフの可視化 ===
def visualize_graph(app):
    """グラフを可視化する
    
    Args:
        app: LangGraphのアプリケーション
    """
    graph = app.get_graph()
    
    print("=" * 50)
    print("【Graph構造】")
    print("=" * 50)
    print(graph.draw_ascii())
    print("=" * 50)
    print()

# === 実行 ===
if __name__ == "__main__":
    # グラフを作成
    app = create_graph()
    
    # グラフを可視化
    visualize_graph(app)

    # 初期状態を渡して実行
    result = app.invoke({
        "question": "パスワードを何回か間違えてログインできなくなりました。どうすればいいですか？"
    })

    print("=" * 50)
    print("【質問】")
    print(result["question"])
    print()
    print("【回答】")
    print(result["answer"])
