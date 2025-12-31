"""
Step 1: 最小エージェント
- LangGraphの基本構造を理解する
- StateGraph, ノード, エッジの使い方を学ぶ
"""

import json
import operator
import sys
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.configs import Settings
from src.models import Plan, ReflectionResult, Subtask, ToolResult
from src.prompts import (
    CREATE_LAST_ANSWER_SYSTEM_PROMPT,
    CREATE_LAST_ANSWER_USER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    SUBTASK_REFLECTION_USER_PROMPT,
    SUBTASK_RETRY_USER_PROMPT,
    SUBTASK_SYSTEM_PROMPT,
)
from src.tools.search_xyz_manual import search_xyz_manual
from src.tools.search_xyz_qa import search_xyz_qa

# .envファイルをos.environに読み込む
load_dotenv()


# === ツールリスト ===
TOOLS = [search_xyz_manual, search_xyz_qa]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# === 状態の定義 ===
# TypedDictを使って、エージェントが持つ状態を定義する
class AgentState(TypedDict):
    question: str  # ユーザーからの質問
    plan: list[str]  # 計画（サブタスクのリスト）
    current_step: int  # 現在のサブタスク番号
    subtask_results: Annotated[Sequence[Subtask], operator.add]  # サブタスクの結果リスト
    answer: str  # エージェントの回答


class SubGraphState(TypedDict):
    """サブグラフの状態"""

    question: str
    plan: list[str]
    subtask: str
    challenge_count: int
    is_completed: bool
    messages: list[ChatCompletionMessageParam]
    tool_results: Annotated[Sequence[ToolResult], operator.add]
    subtask_answer: str


# ==========================================
# サブグラフのノード定義
# ==========================================

# SubGraph内の最大試行回数
MAX_CHALLENGE_COUNT = 3


def select_tools(state: SubGraphState) -> dict:
    """LLMにツールを選択するノード"""

    print(f"  [SubGraph] select_tools: {state['subtask']}")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # LangChainツールをOpenAI形式に変換
    openai_tools = [convert_to_openai_tool(tool) for tool in TOOLS]

    # 初回かリトライかでプロンプトを切り替え
    if state["challenge_count"] == 0:
        user_prompt = f"サブタスク: {state['subtask']}\n\n適切なツールを選択して実行してください。"
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SUBTASK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    else:
        # リトライ時は過去の対話履歴を使用
        messages = list(state["messages"])
        messages.append({"role": "user", "content": SUBTASK_RETRY_USER_PROMPT})

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        tools=openai_tools,
        temperature=0,
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        ai_message: ChatCompletionMessageParam = {
            "role": "assistant",
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        }
        messages.append(ai_message)

    return {"messages": messages}


def execute_tools(state: SubGraphState) -> dict:
    """選択されたツールを実行するノード"""

    print("  [SubGraph] execute_tools")

    messages = list(state["messages"])
    last_message = messages[-1]

    # tool_callsがない場合は空のリストを返す
    tool_calls = last_message.get("tool_calls", [])
    if not tool_calls:
        return {"messages": messages, "tool_results": []}

    tool_results = []

    for tc in tool_calls:
        tool_name = tc["function"]["name"]
        tool_args = tc["function"]["arguments"]

        tool_fn = TOOL_MAP[tool_name]
        result = tool_fn.invoke(tool_args)

        tool_results.append(ToolResult(tool_name=tool_name, args=tool_args, results=result))

        tool_message: ChatCompletionMessageParam = {
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False),
            "tool_call_id": tc["id"],
        }
        messages.append(tool_message)

    return {"messages": messages, "tool_results": tool_results}


def create_subtask_answer(state: SubGraphState) -> dict:
    """サブタスク回答を作成するノード"""

    print("  [SubGraph] create_subtask_answer")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=state["messages"],
        temperature=0,
    )

    subtask_answer = response.choices[0].message.content or ""

    print(f"    回答: {subtask_answer}")

    messages = list(state["messages"])
    messages.append({"role": "assistant", "content": subtask_answer})

    return {"messages": messages, "subtask_answer": subtask_answer}


def reflect_subtask(state: SubGraphState) -> dict:
    """サブタスクを評価するノード"""

    print("  [SubGraph] reflect_subtask")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = list(state["messages"])
    messages.append({"role": "user", "content": SUBTASK_REFLECTION_USER_PROMPT})

    # Structured Outputを使って評価結果を生成
    response = client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=messages,
        response_format=ReflectionResult,
        temperature=0,
    )

    reflection = response.choices[0].message.parsed
    if reflection is None:
        raise ValueError("Reflection result is None")

    messages.append({"role": "assistant", "content": reflection.model_dump_json()})

    print(f"    評価結果: {'OK' if reflection.is_completed else 'NG'}")
    print(f"    アドバイス: {reflection.advice}")

    update = {
        "messages": messages,
        "challenge_count": state["challenge_count"] + 1,
        "is_completed": reflection.is_completed,
    }
    return update


def should_continue_subgraph(state: SubGraphState) -> Literal["end", "continue"]:
    """サブグラフの継続判定"""
    if state["is_completed"] or state["challenge_count"] >= MAX_CHALLENGE_COUNT:
        return "end"
    else:
        return "continue"


def create_subgraph() -> Pregel:
    """サブグラフを作成する"""
    workflow = StateGraph(SubGraphState)

    workflow.add_node("select_tools", select_tools)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("create_subtask_answer", create_subtask_answer)
    workflow.add_node("reflect_subtask", reflect_subtask)

    workflow.add_edge(START, "select_tools")
    workflow.add_edge("select_tools", "execute_tools")
    workflow.add_edge("execute_tools", "create_subtask_answer")
    workflow.add_edge("create_subtask_answer", "reflect_subtask")
    workflow.add_conditional_edges(
        "reflect_subtask",
        should_continue_subgraph,
        {"continue": "select_tools", "end": END},
    )

    return workflow.compile()


# ==========================================
# メイングラフのノード定義
# ==========================================


def create_plan(state: AgentState) -> dict:
    """計画を作成するノード"""

    print("[Node] create_plan")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": PLANNER_USER_PROMPT.format(question=state["question"])},
    ]

    # Structured Outputを使って計画を生成
    response = client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=messages,
        response_format=Plan,
        temperature=0,
    )

    plan = response.choices[0].message.parsed
    if plan is None:
        raise ValueError("Plan is None")

    print(f"  計画: {plan.subtasks}")

    return {"plan": plan.subtasks, "current_step": 0}


def route_subtasks(state: AgentState) -> list[Send]:
    """サブタスクを並列実行するためのルーティング"""

    print(f"[Route] {len(state['plan'])}個のサブタスクを並列実行")

    # 各サブタスクに対してSendを生成
    return [
        Send(
            "execute_subtasks",
            {
                "question": state["question"],
                "plan": state["plan"],
                "current_step": idx,
            },
        )
        for idx, _ in enumerate(state["plan"])
    ]


def execute_subtasks(state: AgentState) -> dict:
    """サブタスクを実行するノード"""

    subtask = state["plan"][state["current_step"]]
    print(f"[Node] execute_subtasks ({state['current_step'] + 1}/{len(state['plan'])}): {subtask}")

    subgraph = create_subgraph()
    result = subgraph.invoke(
        {
            "question": state["question"],
            "plan": state["plan"],
            "subtask": subtask,
            "challenge_count": 0,
            "is_completed": False,
            "messages": [],
            "tool_results": [],
            "subtask_answer": "",
        }
    )

    subtask_result = Subtask(
        task_name=subtask,
        tool_results=list(result["tool_results"]),
        subtask_answer=result["subtask_answer"],
        is_completed=result["is_completed"],
        challenge_count=result["challenge_count"],
    )

    # operator.addで自動マージされる
    return {"subtask_results": [subtask_result]}


def create_answer(state: AgentState) -> dict:
    """最終回答を生成するノード"""

    print("[Node] create_answer")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # サブタスク結果を文字列に変換
    subtask_results_str = "\n\n".join([f"【{r.task_name}】\n{r.subtask_answer}" for r in state["subtask_results"]])

    messages = [
        {"role": "system", "content": CREATE_LAST_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": CREATE_LAST_ANSWER_USER_PROMPT.format(
                question=state["question"], subtask_results=subtask_results_str
            ),
        },
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
    )

    return {"answer": response.choices[0].message.content or ""}


# === グラフの構築 ===
def create_graph() -> Pregel:
    """メイングラフを構築する"""

    workflow = StateGraph(AgentState)

    # ノードを追加
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_subtasks", execute_subtasks)
    workflow.add_node("create_answer", create_answer)

    workflow.add_edge(START, "create_plan")
    workflow.add_conditional_edges(
        "create_plan",
        route_subtasks,
    )

    workflow.add_edge("execute_subtasks", "create_answer")
    workflow.add_edge("create_answer", END)

    return workflow.compile()


# === グラフの可視化 ===
def visualize_graph(app):
    """グラフを可視化する

    Args:
        app: LangGraphのアプリケーション
    """
    # メイングラフの可視化
    graph = app.get_graph()

    print("=" * 50)
    print("【メインGraph構造】")
    print("=" * 50)
    print(graph.draw_mermaid())
    print("=" * 50)
    print()

    # サブグラフの可視化
    subgraph = create_subgraph()
    subgraph_graph = subgraph.get_graph()

    print("=" * 50)
    print("【サブタスク構造】")
    print("=" * 50)
    print(subgraph_graph.draw_mermaid())
    print("=" * 50)
    print()


# === 実行 ===
if __name__ == "__main__":
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
        # Q5: 曖昧なキーワード（セキュリティ→二段階認証、スマホ→認証アプリへの再検索を期待）
        "セキュリティ的なやつを設定したいんですけど、スマホのアプリを使う方法ってありますか？",
        # Q6: Q&Aに回答がない質問（パスワード文字制限の詳細は未登録）
        "パスワードって何文字まで使えますか？あと記号は使えますか？",
    ]

    # 引数で質問番号を指定(optional)。指定なしなら全ての質問を実行
    question_index = sys.argv[1] if len(sys.argv) > 1 else None
    if question_index is not None:
        questions = [questions[int(question_index) - 1]]
    else:
        questions = questions

    # グラフを作成
    app = create_graph()

    # グラフを可視化
    visualize_graph(app)

    def run_agent(question: str):
        result = app.invoke({"question": question})
        print("=" * 50)
        print("【質問】")
        print(result["question"])
        print()
        print("【使用したツールと実行結果】")
        for subtask in result.get("subtask_results", []):
            print(f"  サブタスク: {subtask.task_name}")
            print(f"    試行回数: {subtask.challenge_count}")
            print(f"    評価結果: {'OK' if subtask.is_completed else 'NG'}")
            print(f"    回答: {subtask.subtask_answer}")
            for tr in subtask.tool_results:
                print(f"    - {tr.tool_name}({tr.args})")
                print(f"      結果: {json.dumps(tr.results, ensure_ascii=False, indent=8)}")
        print()
        print("【回答】")
        print(result["answer"])
        print()

    for question in questions:
        run_agent(question)
