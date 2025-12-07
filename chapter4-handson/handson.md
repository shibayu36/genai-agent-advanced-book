# LangGraph AIエージェント ハンズオン

## 概要

このハンズオンでは、LangGraphを使ってAIエージェントを段階的に構築していきます。
最小構成から始めて、機能を少しずつ追加しながら、最終的には「Plan-and-Execute + Reflection」パターンのエージェントを完成させます。

### 学べること
- LangGraphの基本（StateGraph, ノード, エッジ）
- Tool Callingの仕組み
- Plan-and-Executeパターン
- Reflectionによる自己評価とリトライ
- サブグラフと並列実行

### 前提条件
- Python 3.12以上
- Docker および Docker Compose
- OpenAIのAPIキー

---

## 環境構築

### 1. 依存関係のインストール

```bash
cd chapter4-handson
uv sync
```

### 2. 環境変数の設定

`.env` ファイルを作成:

```bash
cp .env.sample .env
```

`.env` を編集してAPIキーを設定:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-2024-08-06
```

### 3. Dockerコンテナの起動

```bash
make start.engine
```

### 4. 検索インデックスの構築

```bash
make create.index
```

これでElasticsearchとQdrantにドキュメントがインデックスされます。

### 5. 動作確認

```bash
# Elasticsearchの確認
curl http://localhost:9200/_cat/indices

# Qdrantの確認
curl http://localhost:6333/collections
```

---

## Step 1: 最小エージェント

### 目標
- LangGraphの最も基本的な使い方を理解する
- StateGraph, ノード, エッジの概念を掴む

### 概念説明

LangGraphは「状態（State）」を持ち、「ノード（Node）」が状態を変更しながら処理を進めるフレームワークです。

```
[START] → [ノードA] → [ノードB] → [END]
            ↓            ↓
         状態を更新    状態を更新
```

**重要な概念:**
- **State**: 処理の途中で保持するデータ（TypedDictで定義）
- **Node**: 状態を受け取り、更新部分を返す関数
- **Edge**: ノード間の接続

---

### Think 1-1: 状態とは何か？

コードを書く前に考えてみてください：

> エージェントが「質問に回答する」タスクを実行するとき、
> 処理の途中で保持しておくべき情報は何でしょうか？

<details>
<summary>ヒント</summary>

- 入力として受け取るもの
- 出力として返すもの
- 途中の処理で生成されるもの

</details>

<details>
<summary>回答例</summary>

最小構成では以下が必要：
- `question`: ユーザーからの質問（入力）
- `answer`: エージェントの回答（出力）

より複雑になると：
- `plan`: 計画
- `context`: 検索で得た情報
- `messages`: 会話履歴
など

</details>

---

### 実装

#### 1. `src/configs.py` を作成

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_api_base: str
    openai_model: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

#### 2. `src/agent.py` を作成

```python
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

    # エッジを追加（ノード間の接続）
    workflow.add_edge(START, "create_answer")  # 開始 → create_answer
    workflow.add_edge("create_answer", END)    # create_answer → 終了

    # グラフをコンパイル
    app = workflow.compile()

    return app


# === 実行 ===
if __name__ == "__main__":
    # グラフを作成
    app = create_graph()

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
```

---

### 実行してみよう

```bash
uv run python -m src.agent
```

---

### Think 1-2: この実装の問題点は？

実行結果を見て考えてみてください：

> このエージェントは、XYZシステム固有の情報を知っていますか？
> 正確な回答ができていますか？

<details>
<summary>回答</summary>

問題点：
- LLMは学習データに含まれない「XYZシステム」の具体的な情報を知らない
- 一般的な回答しかできない
- 社内ドキュメントやQ&Aの情報を参照できない

→ **Tool Calling（RAG）** が必要！

</details>

---

### Step 1 まとめ

学んだこと：
- `StateGraph`: 状態を持つグラフを作成
- `TypedDict`: 状態の型を定義
- `add_node()`: ノード（処理）を追加
- `add_edge()`: ノード間の接続を定義
- `compile()`: グラフを実行可能な形にする
- `invoke()`: グラフを実行

**次のステップ**: Tool Callingを追加して、RAGで情報検索できるようにする

---

## Step 2: Tool Calling追加

### 目標
- OpenAIのTool Calling機能を理解する
- RAGツールを使って情報を検索できるようにする

### 概念説明

Tool Callingの流れ：

```
[質問] → [LLMがツールを選択] → [ツール実行] → [結果を使って回答生成]
```

LLMは「どのツールを使うか」「どんな引数で呼ぶか」を自分で決めます。

---

### Think 2-1: ツールの定義には何が必要？

> LLMがツールを「選択」するためには、どんな情報が必要でしょうか？

<details>
<summary>回答</summary>

- **ツール名**: 識別するための名前
- **説明（description）**: どんな時に使うツールか
- **引数の定義**: どんなパラメータを受け取るか

LLMはこれらの情報を見て「この質問にはこのツールが適切だ」と判断します。

</details>

---

### 実装

#### 1. `src/tools/search_xyz_manual.py` を作成

```python
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
        outputs.append({
            "file_name": hit["_source"]["file_name"],
            "content": hit["_source"]["content"]
        })

    return outputs
```

#### 2. `src/tools/search_xyz_qa.py` を作成

```python
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
    query_vector = (
        openai_client.embeddings.create(input=query, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

    # ベクトル検索を実行
    search_results = qdrant_client.query_points(
        collection_name="documents", query=query_vector, limit=MAX_SEARCH_RESULTS
    ).points

    outputs = []
    for point in search_results:
        if point.payload is None:
            continue
        outputs.append({
            "file_name": point.payload["file_name"],
            "content": point.payload["content"]
        })

    return outputs
```

#### 3. `src/agent.py` を更新

```python
"""
Step 2: Tool Calling追加
- OpenAIのTool Calling機能を使う
- RAGで情報を検索して回答する
"""

import json
from typing import TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from src.configs import Settings
from src.tools.search_xyz_manual import search_xyz_manual
from src.tools.search_xyz_qa import search_xyz_qa


# === ツールの準備 ===
TOOLS = [search_xyz_manual, search_xyz_qa]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# === 状態の定義 ===
class AgentState(TypedDict):
    question: str
    tool_calls: list      # LLMが選択したツール呼び出し
    tool_results: list    # ツールの実行結果
    answer: str


# === ノードの定義 ===

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
            "content": "あなたはXYZシステムのヘルプデスク担当です。質問に答えるために必要なツールを選択してください。"
        },
        {
            "role": "user",
            "content": state["question"]
        }
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
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
            for tc in tool_calls
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

        results.append({
            "tool": tool_name,
            "result": result
        })

    return {"tool_results": results}


def create_answer(state: AgentState) -> dict:
    """検索結果を使って回答を生成するノード"""

    print("[Node] create_answer")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = [
        {
            "role": "system",
            "content": """あなたはXYZシステムのヘルプデスク担当です。
検索結果を参考にして、質問に正確に回答してください。
検索結果に情報がない場合は、その旨を伝えてください。"""
        },
        {
            "role": "user",
            "content": f"""質問: {state["question"]}

検索結果:
{json.dumps(state["tool_results"], ensure_ascii=False, indent=2)}

上記の検索結果を参考に回答してください。"""
        }
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
    )

    return {"answer": response.choices[0].message.content}


# === グラフの構築 ===
def create_graph():
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

    return workflow.compile()


# === 実行 ===
if __name__ == "__main__":
    app = create_graph()

    result = app.invoke({
        "question": "パスワードを何回か間違えてログインできなくなりました。どうすればいいですか？"
    })

    print()
    print("=" * 50)
    print("【質問】")
    print(result["question"])
    print()
    print("【使用したツール】")
    for tc in result["tool_calls"]:
        print(f"  - {tc['name']}({tc['arguments']})")
    print()
    print("【回答】")
    print(result["answer"])
```

---

### 実行してみよう

```bash
uv run python -m src.agent
```

---

### Think 2-2: Step 1との違いは？

> Step 1の回答と Step 2の回答を比較してみてください。
> どのような違いがありますか？

<details>
<summary>観察ポイント</summary>

- Step 2では「システム管理者に連絡」など具体的な手順が含まれているはず
- これは実際のマニュアルから検索した情報に基づいている
- LLMの学習データにない情報でも、RAGで正確に回答できる

</details>

---

### Think 2-3: この実装の限界は？

> 以下の複数トピックの質問を投げたらどうなるでしょう？

```python
question = """
1. 特定のプロジェクトに対してのみ通知を制限する方法について
2. パスワードに利用可能な文字の制限について
"""
```

試してみてください！

<details>
<summary>問題点</summary>

- 1回の検索で複数トピックを扱うのは難しい
- 検索キーワードが曖昧になりがち
- 片方のトピックしか答えられないことがある

→ **Plan（計画）** で質問を分解する必要がある！

</details>

---

### Step 2 まとめ

学んだこと：
- LangChainの`@tool`デコレーター
- `convert_to_openai_tool`でOpenAI形式に変換
- `tool_calls`でLLMがツールを選択
- ツール実行結果を使った回答生成

**次のステップ**: Planを追加して、複雑な質問を分解できるようにする

---

## Step 3: Plan追加（計画→実行→回答）

### 目標
- 複雑な質問を分解する「計画」機能を追加
- Structured Outputを使った構造化された出力
- ループ処理の実装

### 概念説明

Plan-and-Executeパターン：

```
[質問] → [計画生成] → [サブタスク実行] → [回答統合]
              ↓              ↑
         "1. 通知制限について調べる"   │
         "2. パスワード仕様について調べる"
                             │
              ←───ループ────┘
```

---

### Think 3-1: なぜ計画が必要？

> 「計画を立てる」ことで、何が改善されるでしょうか？

<details>
<summary>回答</summary>

1. **複雑な質問の分解**: 複数トピックを個別に処理できる
2. **検索精度の向上**: 各トピックに特化したキーワードで検索
3. **回答の網羅性**: 全てのトピックに確実に回答
4. **透明性**: どのような手順で回答を導いたか明確

</details>

---

### 実装

#### 1. `src/models.py` を作成

```python
from pydantic import BaseModel, Field


class Plan(BaseModel):
    """計画を表すモデル"""
    subtasks: list[str] = Field(
        ...,
        description="質問に回答するために必要なサブタスクのリスト"
    )
```

#### 2. `src/prompts.py` を作成

```python
PLANNER_SYSTEM_PROMPT = """
# 役割
あなたはXYZというシステムのヘルプデスク担当者です。
ユーザーの質問に答えるために以下の指示に従って回答作成の計画を立ててください。

# 絶対に守るべき制約事項
- サブタスクはどんな内容について知りたいのかを具体的かつ詳細に記述すること
- サブタスクは同じ内容を調査しないように重複なく構成すること
- 必要最小限のサブタスクを作成すること

# 例
質問: AとBの違いについて教えて
計画:
- Aとは何かについて調べる
- Bとは何かについて調べる

"""

PLANNER_USER_PROMPT = """
{question}
"""
```

#### 3. `src/agent.py` を更新

```python
"""
Step 3: Plan追加
- 質問を分析してサブタスクに分解
- Structured Outputで計画を構造化
- ループで全サブタスクを実行
"""

import json
from typing import TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from src.configs import Settings
from src.models import Plan
from src.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from src.tools.search_xyz_manual import search_xyz_manual
from src.tools.search_xyz_qa import search_xyz_qa


# === ツールの準備 ===
TOOLS = [search_xyz_manual, search_xyz_qa]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# === 状態の定義 ===
class AgentState(TypedDict):
    question: str
    plan: list[str]           # 計画（サブタスクのリスト）
    current_step: int         # 現在のサブタスク番号
    subtask_results: list     # サブタスクの結果リスト
    answer: str


# === ノードの定義 ===

def create_plan(state: AgentState) -> dict:
    """質問を分析して計画を立てるノード"""

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

    print(f"  計画: {plan.subtasks}")

    return {
        "plan": plan.subtasks,
        "current_step": 0,
        "subtask_results": []
    }


def execute_subtask(state: AgentState) -> dict:
    """1つのサブタスクを実行するノード"""

    current_step = state["current_step"]
    subtask = state["plan"][current_step]

    print(f"[Node] execute_subtask ({current_step + 1}/{len(state['plan'])}): {subtask}")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # LangChainツールをOpenAI形式に変換
    openai_tools = [convert_to_openai_tool(tool) for tool in TOOLS]

    # ツール選択
    messages = [
        {
            "role": "system",
            "content": "サブタスクを実行するために適切なツールを選択してください。"
        },
        {
            "role": "user",
            "content": f"サブタスク: {subtask}"
        }
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        tools=openai_tools,
        temperature=0,
    )

    # ツール実行
    tool_results = []
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for tc in tool_calls:
            tool_fn = TOOL_MAP[tc.function.name]
            result = tool_fn.invoke(tc.function.arguments)
            tool_results.append({
                "tool": tc.function.name,
                "result": result
            })

    # サブタスクの回答生成
    answer_messages = [
        {
            "role": "system",
            "content": "検索結果を参考に、サブタスクに対する回答を作成してください。"
        },
        {
            "role": "user",
            "content": f"""サブタスク: {subtask}

検索結果:
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

回答を作成してください。"""
        }
    ]

    answer_response = client.chat.completions.create(
        model=settings.openai_model,
        messages=answer_messages,
        temperature=0,
    )

    subtask_answer = answer_response.choices[0].message.content

    # 結果を追加
    new_results = state["subtask_results"] + [{
        "subtask": subtask,
        "answer": subtask_answer
    }]

    return {
        "subtask_results": new_results,
        "current_step": current_step + 1
    }


def should_continue(state: AgentState) -> str:
    """全てのサブタスクが完了したかチェック"""

    if state["current_step"] < len(state["plan"]):
        return "continue"
    else:
        return "finish"


def create_answer(state: AgentState) -> dict:
    """全サブタスク結果を統合して最終回答を作成"""

    print("[Node] create_answer")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    subtask_summary = "\n\n".join([
        f"【{r['subtask']}】\n{r['answer']}"
        for r in state["subtask_results"]
    ])

    messages = [
        {
            "role": "system",
            "content": """あなたはXYZシステムのヘルプデスク担当です。
サブタスクの結果をもとに、ユーザーへの最終回答を作成してください。
- 回答は丁寧で分かりやすく
- 質問された全ての項目に回答すること
- 不確定な情報は含めない"""
        },
        {
            "role": "user",
            "content": f"""ユーザーの質問: {state["question"]}

サブタスクの結果:
{subtask_summary}

最終回答を作成してください。"""
        }
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
    )

    return {"answer": response.choices[0].message.content}


# === グラフの構築 ===
def create_graph():
    workflow = StateGraph(AgentState)

    # ノードを追加
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_subtask", execute_subtask)
    workflow.add_node("create_answer", create_answer)

    # エッジを追加
    workflow.add_edge(START, "create_plan")
    workflow.add_edge("create_plan", "execute_subtask")

    # 条件分岐: サブタスクが残っていれば続行、なければ回答作成へ
    workflow.add_conditional_edges(
        "execute_subtask",
        should_continue,
        {
            "continue": "execute_subtask",  # ループ
            "finish": "create_answer"
        }
    )

    workflow.add_edge("create_answer", END)

    return workflow.compile()


# === 実行 ===
if __name__ == "__main__":
    app = create_graph()

    # Q3: 複数トピックの質問
    question = """お世話になっております。
以下の点についてご教示いただければと存じます。

1. 特定のプロジェクトに対してのみ通知を制限する方法について

2. パスワードに利用可能な文字の制限について

よろしくお願いいたします。"""

    result = app.invoke({"question": question})

    print()
    print("=" * 50)
    print("【質問】")
    print(result["question"])
    print()
    print("【計画】")
    for i, task in enumerate(result["plan"], 1):
        print(f"  {i}. {task}")
    print()
    print("【最終回答】")
    print(result["answer"])
```

---

### 実行してみよう

```bash
uv run python -m src.agent
```

---

### Think 3-2: ループの仕組み

> `add_conditional_edges` を使ったループの仕組みを図で描けますか？

<details>
<summary>回答</summary>

```
[START]
   ↓
[create_plan]
   ↓
[execute_subtask] ←─────┐
   ↓                    │
{should_continue}       │
   ├─ "continue" ───────┘
   ↓
 "finish"
   ↓
[create_answer]
   ↓
[END]
```

`should_continue` 関数が「まだサブタスクがある」と判断すると、
`execute_subtask` に戻ってループします。

</details>

---

### Think 3-3: Structured Outputの利点

> なぜ `response_format=Plan` を使うのでしょうか？
> 普通のテキスト出力と何が違う？

<details>
<summary>回答</summary>

**Structured Outputの利点:**
1. **型安全**: Pydanticモデルでバリデーション
2. **パース不要**: JSONをパースする必要がない
3. **確実性**: 必ず指定した形式で返ってくる

普通のテキストだと「箇条書きをパースする」などの処理が必要で、
フォーマットが崩れるリスクがある。

</details>

---

### Step 3 まとめ

学んだこと：
- Plan-and-Executeパターンの基本
- Structured Output（`response_format`）
- `add_conditional_edges` による条件分岐
- ループ処理の実装

**次のステップ**: サブタスク結果の管理を改善する

---

## Step 4: サブタスク分割の改善

### 目標
- `Annotated` と `operator.add` を使った状態の蓄積を理解する
- サブタスク結果をより構造化して管理する

### 概念説明

Step 3では `subtask_results` をリストで管理していましたが、毎回全体を更新する必要がありました。
`Annotated` と `operator.add` を使うと、**追加分だけを返せば自動的にマージ**されます。

```python
# Before: 毎回全体を返す
new_results = state["subtask_results"] + [new_item]
return {"subtask_results": new_results}

# After: 追加分だけ返す
return {"subtask_results": [new_item]}  # 自動的にマージされる
```

---

### Think 4-1: なぜこの仕組みが必要？

> 並列実行を考えたとき、なぜ「追加分だけ返す」仕組みが重要でしょうか？

<details>
<summary>回答</summary>

並列実行では複数のノードが同時に動作します。
もし「全体を置き換える」方式だと、後から完了したノードの結果で上書きされてしまいます。

「追加分をマージ」する方式なら、並列で動作しても全ての結果が保持されます。

</details>

---

### 実装

#### 1. `src/models.py` を更新

```python
from pydantic import BaseModel, Field


class Plan(BaseModel):
    """計画を表すモデル"""
    subtasks: list[str] = Field(
        ...,
        description="質問に回答するために必要なサブタスクのリスト"
    )


class ToolResult(BaseModel):
    """ツール実行結果"""
    tool_name: str = Field(..., description="ツールの名前")
    args: str = Field(..., description="ツールの引数")
    results: list[dict] = Field(..., description="ツールの結果")


class Subtask(BaseModel):
    """サブタスクの実行結果"""
    task_name: str = Field(..., description="サブタスクの名前")
    tool_results: list[ToolResult] = Field(..., description="ツール実行結果")
    subtask_answer: str = Field(..., description="サブタスクの回答")
```

#### 2. `src/agent.py` を更新

```python
"""
Step 4: サブタスク分割の改善
- Annotated + operator.add で状態を蓄積
- サブタスク結果を構造化
"""

import operator
import json
from typing import Annotated, Sequence, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from src.configs import Settings
from src.models import Plan, Subtask, ToolResult
from src.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from src.tools.search_xyz_manual import search_xyz_manual
from src.tools.search_xyz_qa import search_xyz_qa


# === ツールの準備 ===
TOOLS = [search_xyz_manual, search_xyz_qa]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# === 状態の定義 ===
class AgentState(TypedDict):
    question: str
    plan: list[str]
    current_step: int
    # Annotated + operator.add で「追加分だけ返せばマージされる」
    subtask_results: Annotated[Sequence[Subtask], operator.add]
    answer: str


# === ノードの定義 ===

def create_plan(state: AgentState) -> dict:
    """質問を分析して計画を立てるノード"""

    print("[Node] create_plan")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": PLANNER_USER_PROMPT.format(question=state["question"])},
    ]

    response = client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=messages,
        response_format=Plan,
        temperature=0,
    )

    plan = response.choices[0].message.parsed

    print(f"  計画: {plan.subtasks}")

    return {
        "plan": plan.subtasks,
        "current_step": 0,
    }


def execute_subtask(state: AgentState) -> dict:
    """1つのサブタスクを実行するノード"""

    current_step = state["current_step"]
    subtask = state["plan"][current_step]

    print(f"[Node] execute_subtask ({current_step + 1}/{len(state['plan'])}): {subtask}")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    openai_tools = [convert_to_openai_tool(tool) for tool in TOOLS]

    # ツール選択
    messages = [
        {
            "role": "system",
            "content": "サブタスクを実行するために適切なツールを選択してください。"
        },
        {
            "role": "user",
            "content": f"サブタスク: {subtask}"
        }
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        tools=openai_tools,
        temperature=0,
    )

    # ツール実行（構造化された結果）
    tool_results = []
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for tc in tool_calls:
            tool_fn = TOOL_MAP[tc.function.name]
            result = tool_fn.invoke(tc.function.arguments)
            tool_results.append(ToolResult(
                tool_name=tc.function.name,
                args=tc.function.arguments,
                results=result
            ))

    # サブタスクの回答生成
    answer_messages = [
        {
            "role": "system",
            "content": "検索結果を参考に、サブタスクに対する回答を作成してください。"
        },
        {
            "role": "user",
            "content": f"""サブタスク: {subtask}

検索結果:
{json.dumps([tr.model_dump() for tr in tool_results], ensure_ascii=False, indent=2)}

回答を作成してください。"""
        }
    ]

    answer_response = client.chat.completions.create(
        model=settings.openai_model,
        messages=answer_messages,
        temperature=0,
    )

    subtask_answer = answer_response.choices[0].message.content

    # Subtaskオブジェクトを作成
    subtask_result = Subtask(
        task_name=subtask,
        tool_results=tool_results,
        subtask_answer=subtask_answer
    )

    # 追加分だけを返す（operator.addで自動マージ）
    return {
        "subtask_results": [subtask_result],
        "current_step": current_step + 1
    }


def should_continue(state: AgentState) -> str:
    """全てのサブタスクが完了したかチェック"""

    if state["current_step"] < len(state["plan"]):
        return "continue"
    else:
        return "finish"


def create_answer(state: AgentState) -> dict:
    """全サブタスク結果を統合して最終回答を作成"""

    print("[Node] create_answer")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Subtaskオブジェクトから情報を取得
    subtask_summary = "\n\n".join([
        f"【{r.task_name}】\n{r.subtask_answer}"
        for r in state["subtask_results"]
    ])

    messages = [
        {
            "role": "system",
            "content": """あなたはXYZシステムのヘルプデスク担当です。
サブタスクの結果をもとに、ユーザーへの最終回答を作成してください。
- 回答は丁寧で分かりやすく
- 質問された全ての項目に回答すること
- 不確定な情報は含めない"""
        },
        {
            "role": "user",
            "content": f"""ユーザーの質問: {state["question"]}

サブタスクの結果:
{subtask_summary}

最終回答を作成してください。"""
        }
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
    )

    return {"answer": response.choices[0].message.content}


# === グラフの構築 ===
def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_subtask", execute_subtask)
    workflow.add_node("create_answer", create_answer)

    workflow.add_edge(START, "create_plan")
    workflow.add_edge("create_plan", "execute_subtask")

    workflow.add_conditional_edges(
        "execute_subtask",
        should_continue,
        {
            "continue": "execute_subtask",
            "finish": "create_answer"
        }
    )

    workflow.add_edge("create_answer", END)

    return workflow.compile()


# === 実行 ===
if __name__ == "__main__":
    app = create_graph()

    question = """お世話になっております。
以下の点についてご教示いただければと存じます。

1. 特定のプロジェクトに対してのみ通知を制限する方法について

2. パスワードに利用可能な文字の制限について

よろしくお願いいたします。"""

    result = app.invoke({"question": question})

    print()
    print("=" * 50)
    print("【質問】")
    print(result["question"])
    print()
    print("【計画】")
    for i, task in enumerate(result["plan"], 1):
        print(f"  {i}. {task}")
    print()
    print("【サブタスク結果】")
    for r in result["subtask_results"]:
        print(f"  - {r.task_name}")
        print(f"    使用ツール: {[tr.tool_name for tr in r.tool_results]}")
    print()
    print("【最終回答】")
    print(result["answer"])
```

---

### 実行してみよう

```bash
uv run python -m src.agent
```

---

### Think 4-2: Annotated の仕組み

> `Annotated[Sequence[Subtask], operator.add]` は何を意味していますか？

<details>
<summary>回答</summary>

- `Sequence[Subtask]`: 型は「Subtaskのシーケンス（リスト）」
- `operator.add`: 状態を更新するとき、置き換えではなく「追加（+）」で結合

つまり：
```python
# ノードが {"subtask_results": [new_item]} を返すと
# 既存の subtask_results + [new_item] が新しい状態になる
```

</details>

---

### Step 4 まとめ

学んだこと：
- `Annotated` + `operator.add` による状態の蓄積
- Pydanticモデルでの結果構造化
- 並列実行の準備

**次のステップ**: Reflectionを追加して、回答の品質を自己評価する

---

## Step 5: Reflection追加

### 目標
- 自己評価（Reflection）パターンを実装する
- 不十分な回答の場合にリトライする仕組みを作る

### 概念説明

Reflectionパターン：

```
[サブタスク実行] → [回答生成] → [自己評価]
                                   ↓
                              OK? ─┬─ Yes → 次へ
                                   └─ No  → リトライ（最大3回）
```

LLMに「自分の回答が十分か」を評価させ、不十分なら別のアプローチでリトライします。

---

### Think 5-1: なぜ自己評価が必要？

> 自己評価（Reflection）を入れることで、何が改善されますか？

<details>
<summary>回答</summary>

1. **回答の品質向上**: 不十分な回答を検出してやり直せる
2. **検索の改善**: 別のキーワードや別のツールを試せる
3. **robustness**: 最初の検索で情報が見つからなくても諦めない

例えば「情報が見つかりませんでした」という回答を検出して、
別の検索キーワードでリトライできます。

</details>

---

### 実装

#### 1. `src/models.py` を更新

```python
from pydantic import BaseModel, Field


class Plan(BaseModel):
    """計画を表すモデル"""
    subtasks: list[str] = Field(
        ...,
        description="質問に回答するために必要なサブタスクのリスト"
    )


class ToolResult(BaseModel):
    """ツール実行結果"""
    tool_name: str = Field(..., description="ツールの名前")
    args: str = Field(..., description="ツールの引数")
    results: list[dict] = Field(..., description="ツールの結果")


class ReflectionResult(BaseModel):
    """リフレクション（自己評価）の結果"""
    advice: str = Field(
        ...,
        description="評価がNGの場合のアドバイス。別のツールを試す、別の文言で検索するなど。"
    )
    is_completed: bool = Field(
        ...,
        description="サブタスクに対して正しく回答できているかの評価結果"
    )


class Subtask(BaseModel):
    """サブタスクの実行結果"""
    task_name: str = Field(..., description="サブタスクの名前")
    tool_results: list[ToolResult] = Field(..., description="ツール実行結果")
    subtask_answer: str = Field(..., description="サブタスクの回答")
    is_completed: bool = Field(default=True, description="完了フラグ")
    challenge_count: int = Field(default=1, description="試行回数")
```

#### 2. `src/prompts.py` を更新

```python
PLANNER_SYSTEM_PROMPT = """
# 役割
あなたはXYZというシステムのヘルプデスク担当者です。
ユーザーの質問に答えるために以下の指示に従って回答作成の計画を立ててください。

# 絶対に守るべき制約事項
- サブタスクはどんな内容について知りたいのかを具体的かつ詳細に記述すること
- サブタスクは同じ内容を調査しないように重複なく構成すること
- 必要最小限のサブタスクを作成すること

# 例
質問: AとBの違いについて教えて
計画:
- Aとは何かについて調べる
- Bとは何かについて調べる

"""

PLANNER_USER_PROMPT = """
{question}
"""

SUBTASK_SYSTEM_PROMPT = """
あなたはXYZというシステムの質問応答のためにサブタスク実行を担当するエージェントです。
サブタスクはユーザーの質問に回答するために考えられた計画の一つです。

ツールの実行結果から得られた回答に必要なことは言語化してください。
回答できなかった場合は、その旨を言語化してください。
"""

SUBTASK_REFLECTION_USER_PROMPT = """
ツールの実行結果と回答から、サブタスクに対して正しく回答できているかを評価してください。

評価がNGの場合は、別のツールを試す、別の文言でツールを試すなど、
なぜNGなのかとどうしたら改善できるかを考えアドバイスを作成してください。
"""

SUBTASK_RETRY_USER_PROMPT = """
前回の評価結果に従って、別のアプローチでツールを選択・実行してください。
過去に試したキーワードとは異なるキーワードで検索してください。
"""
```

#### 3. `src/agent.py` を更新

```python
"""
Step 5: Reflection追加
- 自己評価パターン
- 条件分岐とリトライ
"""

import operator
import json
from typing import Annotated, Sequence, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.configs import Settings
from src.models import Plan, ReflectionResult, Subtask, ToolResult
from src.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    SUBTASK_SYSTEM_PROMPT,
    SUBTASK_REFLECTION_USER_PROMPT,
    SUBTASK_RETRY_USER_PROMPT,
)
from src.tools.search_xyz_manual import search_xyz_manual
from src.tools.search_xyz_qa import search_xyz_qa


MAX_CHALLENGE_COUNT = 3

# === ツールの準備 ===
TOOLS = [search_xyz_manual, search_xyz_qa]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# === 状態の定義 ===
class AgentState(TypedDict):
    question: str
    plan: list[str]
    current_step: int
    subtask_results: Annotated[Sequence[Subtask], operator.add]
    answer: str


class SubtaskState(TypedDict):
    """サブタスク実行用の内部状態"""
    question: str
    plan: list[str]
    subtask: str
    messages: list[ChatCompletionMessageParam]
    tool_results: list[ToolResult]
    subtask_answer: str
    is_completed: bool
    challenge_count: int


# === ノードの定義 ===

def create_plan(state: AgentState) -> dict:
    """質問を分析して計画を立てるノード"""

    print("[Node] create_plan")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": PLANNER_USER_PROMPT.format(question=state["question"])},
    ]

    response = client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=messages,
        response_format=Plan,
        temperature=0,
    )

    plan = response.choices[0].message.parsed

    print(f"  計画: {plan.subtasks}")

    return {
        "plan": plan.subtasks,
        "current_step": 0,
    }


def execute_subtask_with_reflection(state: AgentState) -> dict:
    """サブタスクをReflection付きで実行"""

    current_step = state["current_step"]
    subtask = state["plan"][current_step]

    print(f"[Node] execute_subtask ({current_step + 1}/{len(state['plan'])}): {subtask}")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)
    openai_tools = [convert_to_openai_tool(tool) for tool in TOOLS]

    # サブタスク用の状態を初期化
    subtask_state: SubtaskState = {
        "question": state["question"],
        "plan": state["plan"],
        "subtask": subtask,
        "messages": [{"role": "system", "content": SUBTASK_SYSTEM_PROMPT}],
        "tool_results": [],
        "subtask_answer": "",
        "is_completed": False,
        "challenge_count": 0,
    }

    while not subtask_state["is_completed"] and subtask_state["challenge_count"] < MAX_CHALLENGE_COUNT:
        subtask_state["challenge_count"] += 1
        print(f"  試行 {subtask_state['challenge_count']}/{MAX_CHALLENGE_COUNT}")

        # リトライ時は別のアプローチを促す
        if subtask_state["challenge_count"] > 1:
            subtask_state["messages"].append({
                "role": "user",
                "content": SUBTASK_RETRY_USER_PROMPT
            })
        else:
            subtask_state["messages"].append({
                "role": "user",
                "content": f"サブタスク: {subtask}\n\n適切なツールを選択して実行してください。"
            })

        # ツール選択
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=subtask_state["messages"],
            tools=openai_tools,
            temperature=0,
        )

        # ツール実行
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            subtask_state["messages"].append({
                "role": "assistant",
                "tool_calls": [tc.model_dump() for tc in tool_calls]
            })

            for tc in tool_calls:
                tool_fn = TOOL_MAP[tc.function.name]
                result = tool_fn.invoke(tc.function.arguments)
                subtask_state["tool_results"].append(ToolResult(
                    tool_name=tc.function.name,
                    args=tc.function.arguments,
                    results=result
                ))
                subtask_state["messages"].append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "tool_call_id": tc.id
                })

        # サブタスク回答生成
        answer_response = client.chat.completions.create(
            model=settings.openai_model,
            messages=subtask_state["messages"],
            temperature=0,
        )
        subtask_state["subtask_answer"] = answer_response.choices[0].message.content
        subtask_state["messages"].append({
            "role": "assistant",
            "content": subtask_state["subtask_answer"]
        })

        # Reflection（自己評価）
        subtask_state["messages"].append({
            "role": "user",
            "content": SUBTASK_REFLECTION_USER_PROMPT
        })

        reflection_response = client.beta.chat.completions.parse(
            model=settings.openai_model,
            messages=subtask_state["messages"],
            response_format=ReflectionResult,
            temperature=0,
        )

        reflection = reflection_response.choices[0].message.parsed
        subtask_state["is_completed"] = reflection.is_completed

        if reflection.is_completed:
            print(f"    ✓ 評価OK")
        else:
            print(f"    ✗ 評価NG: {reflection.advice}")
            subtask_state["messages"].append({
                "role": "assistant",
                "content": f"評価: NG\nアドバイス: {reflection.advice}"
            })

    # 最大試行回数に達した場合
    if not subtask_state["is_completed"]:
        subtask_state["subtask_answer"] = f"{subtask}の回答が見つかりませんでした。"

    # 結果を作成
    subtask_result = Subtask(
        task_name=subtask,
        tool_results=subtask_state["tool_results"],
        subtask_answer=subtask_state["subtask_answer"],
        is_completed=subtask_state["is_completed"],
        challenge_count=subtask_state["challenge_count"]
    )

    return {
        "subtask_results": [subtask_result],
        "current_step": current_step + 1
    }


def should_continue(state: AgentState) -> str:
    """全てのサブタスクが完了したかチェック"""

    if state["current_step"] < len(state["plan"]):
        return "continue"
    else:
        return "finish"


def create_answer(state: AgentState) -> dict:
    """全サブタスク結果を統合して最終回答を作成"""

    print("[Node] create_answer")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    subtask_summary = "\n\n".join([
        f"【{r.task_name}】\n{r.subtask_answer}"
        for r in state["subtask_results"]
    ])

    messages = [
        {
            "role": "system",
            "content": """あなたはXYZシステムのヘルプデスク担当です。
サブタスクの結果をもとに、ユーザーへの最終回答を作成してください。
- 回答は丁寧で分かりやすく
- 質問された全ての項目に回答すること
- 不確定な情報は含めない"""
        },
        {
            "role": "user",
            "content": f"""ユーザーの質問: {state["question"]}

サブタスクの結果:
{subtask_summary}

最終回答を作成してください。"""
        }
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
    )

    return {"answer": response.choices[0].message.content}


# === グラフの構築 ===
def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_subtask", execute_subtask_with_reflection)
    workflow.add_node("create_answer", create_answer)

    workflow.add_edge(START, "create_plan")
    workflow.add_edge("create_plan", "execute_subtask")

    workflow.add_conditional_edges(
        "execute_subtask",
        should_continue,
        {
            "continue": "execute_subtask",
            "finish": "create_answer"
        }
    )

    workflow.add_edge("create_answer", END)

    return workflow.compile()


# === 実行 ===
if __name__ == "__main__":
    app = create_graph()

    question = """お世話になっております。
以下の点についてご教示いただければと存じます。

1. 特定のプロジェクトに対してのみ通知を制限する方法について

2. パスワードに利用可能な文字の制限について

よろしくお願いいたします。"""

    result = app.invoke({"question": question})

    print()
    print("=" * 50)
    print("【質問】")
    print(result["question"])
    print()
    print("【計画】")
    for i, task in enumerate(result["plan"], 1):
        print(f"  {i}. {task}")
    print()
    print("【サブタスク結果】")
    for r in result["subtask_results"]:
        status = "✓" if r.is_completed else "✗"
        print(f"  {status} {r.task_name} (試行: {r.challenge_count}回)")
    print()
    print("【最終回答】")
    print(result["answer"])
```

---

### 実行してみよう

```bash
uv run python -m src.agent
```

---

### Think 5-2: Reflectionのトレードオフ

> Reflectionを入れることのデメリットは何でしょうか？

<details>
<summary>回答</summary>

1. **レイテンシの増加**: 評価のためのLLM呼び出しが追加される
2. **コストの増加**: API呼び出し回数が増える
3. **無限ループのリスク**: 適切な終了条件（MAX_CHALLENGE_COUNT）が必要

トレードオフを考慮して、本当に必要な場面でのみReflectionを使うことが重要。

</details>

---

### Step 5 まとめ

学んだこと：
- Reflectionパターンの実装
- whileループによるリトライ
- 会話履歴（messages）の管理
- 最大試行回数による終了条件

**次のステップ**: 並列実行でパフォーマンスを向上させる

---

## Step 6: 並列実行

### 目標
- LangGraphの`Send`機能を使って並列実行を実装する
- 複数サブタスクを同時に処理してパフォーマンスを向上させる

### 概念説明

Step 5までは直列実行：
```
[Plan] → [Task1] → [Task2] → [Task3] → [Answer]
```

並列実行では：
```
[Plan] → [Task1] ─┐
         [Task2] ─┼→ [Answer]
         [Task3] ─┘
```

`Send` を使うと、複数のノードを同時に起動できます。

---

### Think 6-1: なぜ並列実行が有効？

> 並列実行のメリットとデメリットを考えてみてください。

<details>
<summary>回答</summary>

**メリット:**
- 処理時間の短縮（3タスクが同時に実行される）
- 効率的なリソース利用

**デメリット:**
- 実装が複雑になる
- デバッグが難しくなる
- タスク間で依存関係がある場合は使えない

今回のサブタスクは独立しているので、並列実行が有効です。

</details>

---

### 実装

#### `src/agent.py` を更新

```python
"""
Step 6: 並列実行
- Sendによる並列実行
- パフォーマンス向上
"""

import operator
import json
from typing import Annotated, Sequence, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.configs import Settings
from src.models import Plan, ReflectionResult, Subtask, ToolResult
from src.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    SUBTASK_SYSTEM_PROMPT,
    SUBTASK_REFLECTION_USER_PROMPT,
    SUBTASK_RETRY_USER_PROMPT,
)
from src.tools.search_xyz_manual import search_xyz_manual
from src.tools.search_xyz_qa import search_xyz_qa


MAX_CHALLENGE_COUNT = 3

# === ツールの準備 ===
TOOLS = [search_xyz_manual, search_xyz_qa]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# === 状態の定義 ===
class AgentState(TypedDict):
    question: str
    plan: list[str]
    subtask_results: Annotated[Sequence[Subtask], operator.add]
    answer: str


class SubtaskState(TypedDict):
    """サブタスク実行用の内部状態（並列実行用）"""
    question: str
    plan: list[str]
    subtask: str
    subtask_index: int


# === ノードの定義 ===

def create_plan(state: AgentState) -> dict:
    """質問を分析して計画を立てるノード"""

    print("[Node] create_plan")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": PLANNER_USER_PROMPT.format(question=state["question"])},
    ]

    response = client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=messages,
        response_format=Plan,
        temperature=0,
    )

    plan = response.choices[0].message.parsed

    print(f"  計画: {plan.subtasks}")

    return {"plan": plan.subtasks}


def route_subtasks(state: AgentState) -> list[Send]:
    """サブタスクを並列実行するためのルーティング"""

    print(f"[Route] {len(state['plan'])}個のサブタスクを並列実行")

    # 各サブタスクに対してSendを生成
    return [
        Send(
            "execute_subtask",
            {
                "question": state["question"],
                "plan": state["plan"],
                "subtask": subtask,
                "subtask_index": idx,
            }
        )
        for idx, subtask in enumerate(state["plan"])
    ]


def execute_subtask(state: SubtaskState) -> dict:
    """サブタスクをReflection付きで実行（並列実行対応）"""

    subtask = state["subtask"]
    idx = state["subtask_index"]

    print(f"[Node] execute_subtask ({idx + 1}): {subtask}")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)
    openai_tools = [convert_to_openai_tool(tool) for tool in TOOLS]

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SUBTASK_SYSTEM_PROMPT}
    ]
    tool_results = []
    subtask_answer = ""
    is_completed = False
    challenge_count = 0

    while not is_completed and challenge_count < MAX_CHALLENGE_COUNT:
        challenge_count += 1
        print(f"  [{idx + 1}] 試行 {challenge_count}/{MAX_CHALLENGE_COUNT}")

        if challenge_count > 1:
            messages.append({
                "role": "user",
                "content": SUBTASK_RETRY_USER_PROMPT
            })
        else:
            messages.append({
                "role": "user",
                "content": f"サブタスク: {subtask}\n\n適切なツールを選択して実行してください。"
            })

        # ツール選択
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            tools=openai_tools,
            temperature=0,
        )

        # ツール実行
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            messages.append({
                "role": "assistant",
                "tool_calls": [tc.model_dump() for tc in tool_calls]
            })

            for tc in tool_calls:
                tool_fn = TOOL_MAP[tc.function.name]
                result = tool_fn.invoke(tc.function.arguments)
                tool_results.append(ToolResult(
                    tool_name=tc.function.name,
                    args=tc.function.arguments,
                    results=result
                ))
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "tool_call_id": tc.id
                })

        # サブタスク回答生成
        answer_response = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0,
        )
        subtask_answer = answer_response.choices[0].message.content
        messages.append({
            "role": "assistant",
            "content": subtask_answer
        })

        # Reflection
        messages.append({
            "role": "user",
            "content": SUBTASK_REFLECTION_USER_PROMPT
        })

        reflection_response = client.beta.chat.completions.parse(
            model=settings.openai_model,
            messages=messages,
            response_format=ReflectionResult,
            temperature=0,
        )

        reflection = reflection_response.choices[0].message.parsed
        is_completed = reflection.is_completed

        if is_completed:
            print(f"    [{idx + 1}] ✓ 評価OK")
        else:
            print(f"    [{idx + 1}] ✗ 評価NG: {reflection.advice}")
            messages.append({
                "role": "assistant",
                "content": f"評価: NG\nアドバイス: {reflection.advice}"
            })

    if not is_completed:
        subtask_answer = f"{subtask}の回答が見つかりませんでした。"

    subtask_result = Subtask(
        task_name=subtask,
        tool_results=tool_results,
        subtask_answer=subtask_answer,
        is_completed=is_completed,
        challenge_count=challenge_count
    )

    # operator.addで自動マージされる
    return {"subtask_results": [subtask_result]}


def create_answer(state: AgentState) -> dict:
    """全サブタスク結果を統合して最終回答を作成"""

    print("[Node] create_answer")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    subtask_summary = "\n\n".join([
        f"【{r.task_name}】\n{r.subtask_answer}"
        for r in state["subtask_results"]
    ])

    messages = [
        {
            "role": "system",
            "content": """あなたはXYZシステムのヘルプデスク担当です。
サブタスクの結果をもとに、ユーザーへの最終回答を作成してください。
- 回答は丁寧で分かりやすく
- 質問された全ての項目に回答すること
- 不確定な情報は含めない"""
        },
        {
            "role": "user",
            "content": f"""ユーザーの質問: {state["question"]}

サブタスクの結果:
{subtask_summary}

最終回答を作成してください。"""
        }
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
    )

    return {"answer": response.choices[0].message.content}


# === グラフの構築 ===
def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_subtask", execute_subtask)
    workflow.add_node("create_answer", create_answer)

    workflow.add_edge(START, "create_plan")

    # 条件分岐でSendを返すと並列実行される
    workflow.add_conditional_edges(
        "create_plan",
        route_subtasks,
    )

    workflow.add_edge("execute_subtask", "create_answer")

    workflow.set_finish_point("create_answer")

    return workflow.compile()


# === 実行 ===
if __name__ == "__main__":
    app = create_graph()

    question = """お世話になっております。
以下の点についてご教示いただければと存じます。

1. 特定のプロジェクトに対してのみ通知を制限する方法について

2. パスワードに利用可能な文字の制限について

よろしくお願いいたします。"""

    result = app.invoke({"question": question})

    print()
    print("=" * 50)
    print("【質問】")
    print(result["question"])
    print()
    print("【計画】")
    for i, task in enumerate(result["plan"], 1):
        print(f"  {i}. {task}")
    print()
    print("【サブタスク結果】")
    for r in result["subtask_results"]:
        status = "✓" if r.is_completed else "✗"
        print(f"  {status} {r.task_name} (試行: {r.challenge_count}回)")
    print()
    print("【最終回答】")
    print(result["answer"])
```

---

### 実行してみよう

```bash
uv run python -m src.agent
```

---

### Think 6-2: Sendの仕組み

> `Send` を返すと何が起きますか？

<details>
<summary>回答</summary>

`Send` は「このノードをこの状態で実行せよ」という指示です。

```python
Send("execute_subtask", {"subtask": "タスク1", ...})
```

リストで複数の `Send` を返すと、それらが並列に実行されます。
各 `Send` は独立した状態を持ち、結果は `operator.add` でマージされます。

</details>

---

### Step 6 まとめ

学んだこと：
- `Send` による並列実行
- `add_conditional_edges` で `Send` のリストを返す
- `set_finish_point` の使い方

**次のステップ**: サブグラフ化してコードを整理する

---

## Step 7: サブグラフ化

### 目標
- サブタスク実行部分を独立したサブグラフに分離する
- コードの責務分離と再利用性を向上させる

### 概念説明

サブグラフはグラフの中にグラフを入れる仕組みです：

```
[メイングラフ]
  ├─ create_plan
  ├─ [サブグラフ] ←── 独立したグラフとして定義
  │     ├─ select_tools
  │     ├─ execute_tools
  │     ├─ create_subtask_answer
  │     └─ reflect_subtask
  └─ create_answer
```

---

### Think 7-1: サブグラフ化のメリット

> サブグラフに分離することで、何が良くなりますか？

<details>
<summary>回答</summary>

1. **責務の分離**: メイングラフとサブタスク実行ロジックが分離される
2. **テストしやすさ**: サブグラフを単独でテストできる
3. **再利用性**: サブグラフを他のエージェントで再利用できる
4. **可読性**: コードの見通しが良くなる

chapter4の完成形では、サブタスク実行がサブグラフとして実装されています。

</details>

---

### 実装

これが最終形です。chapter4とほぼ同等の構成になります。

#### `src/agent.py` を更新

```python
"""
Step 7: サブグラフ化
- サブタスク実行をサブグラフに分離
- コードの責務分離と再利用性向上
"""

import operator
import json
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.configs import Settings
from src.models import Plan, ReflectionResult, Subtask, ToolResult
from src.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    SUBTASK_SYSTEM_PROMPT,
    SUBTASK_REFLECTION_USER_PROMPT,
    SUBTASK_RETRY_USER_PROMPT,
)
from src.tools.search_xyz_manual import search_xyz_manual
from src.tools.search_xyz_qa import search_xyz_qa


MAX_CHALLENGE_COUNT = 3

# === ツールの準備 ===
TOOLS = [search_xyz_manual, search_xyz_qa]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# === 状態の定義 ===
class AgentState(TypedDict):
    """メイングラフの状態"""
    question: str
    plan: list[str]
    current_step: int
    subtask_results: Annotated[Sequence[Subtask], operator.add]
    answer: str


class SubGraphState(TypedDict):
    """サブグラフの状態"""
    question: str
    plan: list[str]
    subtask: str
    is_completed: bool
    messages: list[ChatCompletionMessageParam]
    challenge_count: int
    tool_results: Annotated[Sequence[ToolResult], operator.add]
    subtask_answer: str


# ==========================================
# サブグラフのノード定義
# ==========================================

def select_tools(state: SubGraphState) -> dict:
    """ツールを選択するノード"""

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)
    openai_tools = [convert_to_openai_tool(tool) for tool in TOOLS]

    if state["challenge_count"] == 0:
        user_prompt = f"サブタスク: {state['subtask']}\n\n適切なツールを選択してください。"
        messages = [
            {"role": "system", "content": SUBTASK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = state["messages"]
        messages.append({"role": "user", "content": SUBTASK_RETRY_USER_PROMPT})

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        tools=openai_tools,
        temperature=0,
    )

    if response.choices[0].message.tool_calls:
        ai_message = {
            "role": "assistant",
            "tool_calls": [tc.model_dump() for tc in response.choices[0].message.tool_calls],
        }
        messages.append(ai_message)

    return {"messages": messages}


def execute_tools(state: SubGraphState) -> dict:
    """ツールを実行するノード"""

    messages = state["messages"]
    tool_calls = messages[-1].get("tool_calls", [])

    tool_results = []

    for tc in tool_calls:
        tool_name = tc["function"]["name"]
        tool_args = tc["function"]["arguments"]

        tool_fn = TOOL_MAP[tool_name]
        result = tool_fn.invoke(tool_args)

        tool_results.append(ToolResult(
            tool_name=tool_name,
            args=tool_args,
            results=result
        ))

        messages.append({
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False),
            "tool_call_id": tc["id"]
        })

    return {"messages": messages, "tool_results": tool_results}


def create_subtask_answer(state: SubGraphState) -> dict:
    """サブタスク回答を作成するノード"""

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=state["messages"],
        temperature=0,
    )

    subtask_answer = response.choices[0].message.content

    messages = state["messages"]
    messages.append({"role": "assistant", "content": subtask_answer})

    return {"messages": messages, "subtask_answer": subtask_answer}


def reflect_subtask(state: SubGraphState) -> dict:
    """サブタスク回答を内省するノード"""

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = state["messages"]
    messages.append({"role": "user", "content": SUBTASK_REFLECTION_USER_PROMPT})

    response = client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=messages,
        response_format=ReflectionResult,
        temperature=0,
    )

    reflection = response.choices[0].message.parsed

    messages.append({
        "role": "assistant",
        "content": reflection.model_dump_json()
    })

    update = {
        "messages": messages,
        "challenge_count": state["challenge_count"] + 1,
        "is_completed": reflection.is_completed,
    }

    if update["challenge_count"] >= MAX_CHALLENGE_COUNT and not reflection.is_completed:
        update["subtask_answer"] = f"{state['subtask']}の回答が見つかりませんでした。"

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
        {"continue": "select_tools", "end": END}
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

    response = client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=messages,
        response_format=Plan,
        temperature=0,
    )

    plan = response.choices[0].message.parsed

    print(f"  計画: {plan.subtasks}")

    return {"plan": plan.subtasks}


def execute_subtasks(state: AgentState) -> dict:
    """サブグラフを実行するノード"""

    subtask = state["plan"][state["current_step"]]
    print(f"[Node] execute_subtask: {subtask}")

    subgraph = create_subgraph()

    result = subgraph.invoke({
        "question": state["question"],
        "plan": state["plan"],
        "subtask": subtask,
        "is_completed": False,
        "challenge_count": 0,
        "messages": [],
        "tool_results": [],
        "subtask_answer": "",
    })

    subtask_result = Subtask(
        task_name=subtask,
        tool_results=list(result["tool_results"]),
        subtask_answer=result["subtask_answer"],
        is_completed=result["is_completed"],
        challenge_count=result["challenge_count"]
    )

    return {"subtask_results": [subtask_result]}


def route_subtasks(state: AgentState) -> list[Send]:
    """サブタスクを並列実行するためのルーティング"""

    print(f"[Route] {len(state['plan'])}個のサブタスクを並列実行")

    return [
        Send(
            "execute_subtasks",
            {
                "question": state["question"],
                "plan": state["plan"],
                "current_step": idx,
            }
        )
        for idx, _ in enumerate(state["plan"])
    ]


def create_answer(state: AgentState) -> dict:
    """最終回答を作成するノード"""

    print("[Node] create_answer")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    subtask_summary = "\n\n".join([
        f"【{r.task_name}】\n{r.subtask_answer}"
        for r in state["subtask_results"]
    ])

    messages = [
        {
            "role": "system",
            "content": """あなたはXYZシステムのヘルプデスク担当です。
サブタスクの結果をもとに、ユーザーへの最終回答を作成してください。
- 回答は丁寧で分かりやすく
- 質問された全ての項目に回答すること
- 不確定な情報は含めない"""
        },
        {
            "role": "user",
            "content": f"""ユーザーの質問: {state["question"]}

サブタスクの結果:
{subtask_summary}

最終回答を作成してください。"""
        }
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
    )

    return {"answer": response.choices[0].message.content}


# === グラフの構築 ===
def create_graph() -> Pregel:
    """メイングラフを作成する"""
    workflow = StateGraph(AgentState)

    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_subtasks", execute_subtasks)
    workflow.add_node("create_answer", create_answer)

    workflow.add_edge(START, "create_plan")

    workflow.add_conditional_edges(
        "create_plan",
        route_subtasks,
    )

    workflow.add_edge("execute_subtasks", "create_answer")

    workflow.set_finish_point("create_answer")

    return workflow.compile()


# === 実行 ===
if __name__ == "__main__":
    app = create_graph()

    question = """お世話になっております。
以下の点についてご教示いただければと存じます。

1. 特定のプロジェクトに対してのみ通知を制限する方法について

2. パスワードに利用可能な文字の制限について

よろしくお願いいたします。"""

    result = app.invoke({"question": question})

    print()
    print("=" * 50)
    print("【質問】")
    print(result["question"])
    print()
    print("【計画】")
    for i, task in enumerate(result["plan"], 1):
        print(f"  {i}. {task}")
    print()
    print("【サブタスク結果】")
    for r in result["subtask_results"]:
        status = "✓" if r.is_completed else "✗"
        print(f"  {status} {r.task_name} (試行: {r.challenge_count}回)")
    print()
    print("【最終回答】")
    print(result["answer"])
```

---

### 実行してみよう

```bash
uv run python -m src.agent
```

---

### Think 7-2: chapter4との比較

> chapter4の実装と比較して、違いを確認してみてください。
> 構造は同じになっていますか？

<details>
<summary>確認ポイント</summary>

chapter4の `agent.py` と比較すると：
- メイングラフとサブグラフの分離 ✓
- `AgentState` と `SubGraphState` の分離 ✓
- `Send` による並列実行 ✓
- Reflection付きサブタスク実行 ✓

chapter4にはカスタムロガーなどの追加機能がありますが、
コアとなるエージェントの構造は同じです。

</details>

---

### Step 7 まとめ

学んだこと：
- サブグラフの作成（`StateGraph` → `compile()`）
- メイングラフからサブグラフを呼び出す
- 責務の分離による可読性・保守性の向上

---

## 完成！

おめでとうございます！ 🎉

7つのステップを通じて、以下を学びました：

1. **Step 1**: LangGraphの基本（StateGraph, ノード, エッジ）
2. **Step 2**: Tool Callingの仕組み
3. **Step 3**: Plan-and-Executeパターン
4. **Step 4**: 状態の蓄積（Annotated + operator.add）
5. **Step 5**: Reflectionによる自己評価
6. **Step 6**: 並列実行（Send）
7. **Step 7**: サブグラフ化

これで、chapter4と同等のAIエージェントを自分で構築できるようになりました！

### 次のチャレンジ

- プロンプトを改善して回答品質を向上させる
- 別のツール（例：Webスクレイピング、DB検索）を追加する
- ストリーミング対応を実装する
- エラーハンドリングを強化する
