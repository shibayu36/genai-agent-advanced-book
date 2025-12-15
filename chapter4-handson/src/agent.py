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
    answer: str  # エージェントの回答


# === ノードの定義 ===
# ノードは「状態を受け取り、更新された状態を返す関数」


def create_answer(state: AgentState) -> dict:
    """質問に対する回答を生成するノード"""

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = [
        {"role": "system", "content": "あなたはXYZシステムのヘルプデスク担当です。質問に簡潔に回答してください。"},
        {"role": "user", "content": state["question"]},
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
    workflow.add_edge("create_answer", END)  # create_answer → 終了

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

    questions = [
        # Q1: ログインロック
        "パスワードを何回か間違えてログインできなくなりました。どうすればいいですか？",
        # Q2: 二段階認証（複数質問）
        """二段階認証を設定したいのですが、認証アプリでの設定方法を教えてください。
また、二段階認証がうまく動作しない場合の対処法も知りたいです。""",
        # Q3: 通知制限・パスワード制限（丁寧な問い合わせ形式）
        """お世話になっております。
現在、XYZシステムの利用を検討しており、以下の点についてご教示いただければと存じます。

1. 特定のプロジェクトに対してのみ通知を制限する方法について

2. パスワードに利用可能な文字の制限について
該システムにてパスワードを設定する際、使用可能な文字の範囲（例：英数字、記号、文字数制限など）について詳しい情報をいただけますでしょうか。

お忙しいところ恐縮ですが、ご対応のほどよろしくお願い申し上げます。""",
        # Q4: パフォーマンス・権限・レポート機能
        """XYZシステムについていくつか質問があります。

1. システムの動作が最近遅くなっているのですが、考えられる原因と対処法を教えてください。

2. レポート作成機能が画面に表示されません。権限の問題でしょうか？確認方法を教えてください。

3. 大量のデータでレポートを作成するとフリーズすることがあります。効率的な作成方法はありますか？

よろしくお願いします。""",
    ]

    # 初期状態を渡して実行
    for question in questions:
        result = app.invoke({"question": question})
        print("=" * 50)
        print("【質問】")
        print(result["question"])
        print()
        print("【回答】")
        print(result["answer"])
        print()
