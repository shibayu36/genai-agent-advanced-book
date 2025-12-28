"""
Step 1: 最小エージェント
- LangGraphの基本構造を理解する
- StateGraph, ノード, エッジの使い方を学ぶ
"""

import json
import operator
import sys
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.configs import Settings
from src.models import Plan, Subtask, ToolResult
from src.prompts import (
    CREATE_LAST_ANSWER_SYSTEM_PROMPT,
    CREATE_LAST_ANSWER_USER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    SUBTASK_SYSTEM_PROMPT,
)
from src.tools.search_xyz_manual import search_xyz_manual
from src.tools.search_xyz_qa import search_xyz_qa

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
    messages: list[ChatCompletionMessageParam]
    tool_results: Annotated[Sequence[ToolResult], operator.add]
    subtask_answer: str


# ==========================================
# サブグラフのノード定義
# ==========================================


def select_tools(state: SubGraphState) -> dict:
    """LLMにツールを選択するノード"""

    print(f"  [SubGraph] select_tools: {state['subtask']}")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # LangChainツールをOpenAI形式に変換
    openai_tools = [convert_to_openai_tool(tool) for tool in TOOLS]

    user_prompt = f"サブタスク: {state['subtask']}\n\n適切なツールを選択して実行してください。"
    messages = [
        {"role": "system", "content": SUBTASK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

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

    return {"subtask_answer": subtask_answer}


def create_subgraph() -> Pregel:
    """サブグラフを作成する"""
    workflow = StateGraph(SubGraphState)

    workflow.add_node("select_tools", select_tools)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("create_subtask_answer", create_subtask_answer)

    workflow.add_edge(START, "select_tools")
    workflow.add_edge("select_tools", "execute_tools")
    workflow.add_edge("execute_tools", "create_subtask_answer")
    workflow.add_edge("create_subtask_answer", END)

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
            "messages": [],
            "tool_results": [],
            "subtask_answer": "",
        }
    )

    subtask_result = Subtask(
        task_name=subtask,
        tool_results=list(result["tool_results"]),
        subtask_answer=result["subtask_answer"],
    )

    # operator.addで自動マージされる
    return {"subtask_results": [subtask_result], "current_step": state["current_step"] + 1}


def should_continue(state: AgentState) -> Literal["continue", "finish"]:
    """全てのサブタスクが完了したかチェック"""
    if state["current_step"] < len(state["plan"]):
        return "continue"
    else:
        return "finish"


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
    workflow.add_edge("create_plan", "execute_subtasks")

    # 条件分岐: サブタスクが残っていれば続行、なければ回答作成へ
    workflow.add_conditional_edges(
        "execute_subtasks", should_continue, {"continue": "execute_subtasks", "finish": "create_answer"}
    )

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
            for tr in subtask.tool_results:
                print(f"    - {tr.tool_name}({tr.args})")
                print(f"      結果: {json.dumps(tr.results, ensure_ascii=False, indent=8)}")
        print()
        print("【回答】")
        print(result["answer"])
        print()

    for question in questions:
        run_agent(question)
