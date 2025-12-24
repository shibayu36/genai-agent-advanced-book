"""
Step 1: 最小エージェント
- LangGraphの基本構造を理解する
- StateGraph, ノード, エッジの使い方を学ぶ
"""

import json
from typing import TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from src.configs import Settings
from src.tools.search_xyz_manual import search_xyz_manual
from src.tools.search_xyz_qa import search_xyz_qa

# === ツールリスト ===
TOOLS = [search_xyz_manual, search_xyz_qa]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# === 状態の定義 ===
# TypedDictを使って、エージェントが持つ状態を定義する
class AgentState(TypedDict):
    question: str  # ユーザーからの質問
    answer: str  # エージェントの回答
    tool_calls: list  # 選択されたツール呼び出し
    tool_results: list  # ツール実行結果


# === ノードの定義 ===
# ノードは「状態を受け取り、更新された状態を返す関数」


def select_tools(state: AgentState) -> dict:
    """LLMにツールを選択させるノード"""

    print("[Node] select_tools")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # LangChainツールをOpenAI形式に変換
    openai_tools = [convert_to_openai_tool(tool) for tool in TOOLS]

    messages = [
        {
            "role": "system",
            "content": "あなたはXYZシステムのヘルプデスク担当です。質問に答えるために必要なツールを選択してください。",
        },
        {"role": "user", "content": state["question"]},
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        tools=openai_tools,
        temperature=0,
    )

    tool_calls = response.choices[0].message.tool_calls

    if tool_calls is None:
        return {"tool_calls": []}

    # tool_callsをシリアライズ可能な形式に変換
    return {
        "tool_calls": [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments} for tc in tool_calls
        ]
    }


def execute_tools(state: AgentState) -> dict:
    """選択されたツールを実行するノード"""

    print("[Node] execute_tools")

    results = []

    for tool_call in state["tool_calls"]:
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]

        # ツールを実行
        tool_fn = TOOL_MAP[tool_name]
        result = tool_fn.invoke(tool_args)

        results.append({"tool": tool_name, "result": result})

    return {"tool_results": results}


def create_answer(state: AgentState) -> dict:
    """質問に対する回答を生成するノード"""

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = [
        {
            "role": "system",
            "content": """あなたはXYZシステムのヘルプデスク担当です。
検索結果を参考にして、質問に正確に回答してください。
検索結果に情報がない場合は、その旨を伝えてください。""",
        },
        {
            "role": "user",
            "content": f"""質問: {state["question"]}

検索結果:
{json.dumps(state["tool_results"], ensure_ascii=False, indent=2)}

上記の検索結果を参考に回答してください。""",
        },
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
    workflow.add_node("select_tools", select_tools)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("create_answer", create_answer)

    # エッジを追加（直列に接続）
    workflow.add_edge(START, "select_tools")
    workflow.add_edge("select_tools", "execute_tools")
    workflow.add_edge("execute_tools", "create_answer")
    workflow.add_edge("create_answer", END)

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
        print("【使用したツールと実行結果】")
        for i, tc in enumerate(result["tool_calls"]):
            print(f"  - {tc['name']}({tc['arguments']})")
            if i < len(result["tool_results"]):
                tr = result["tool_results"][i]
                print(f"    結果: {json.dumps(tr['result'], ensure_ascii=False, indent=6)}")
        print()
        print("【回答】")
        print(result["answer"])
        print()
