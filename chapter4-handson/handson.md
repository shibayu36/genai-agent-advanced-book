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
- サブグラフによる責務分離
- `Annotated` + `operator.add` による状態の蓄積

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

このStepでは、以下の設計を最初から取り入れます：

1. **Subtaskモデル**: サブタスク結果を構造化して管理
2. **Annotated + operator.add**: 追加分だけ返せば自動マージ
3. **サブグラフ**: サブタスク実行ロジックを独立したグラフに分離

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

### Think 3-2: サブグラフとは？

> なぜサブタスク実行を「サブグラフ」として分離するのでしょうか？

<details>
<summary>回答</summary>

1. **責務の分離**: メイングラフとサブタスク実行ロジックが分離される
2. **テストしやすさ**: サブグラフを単独でテストできる
3. **再利用性**: サブグラフを他のエージェントで再利用できる
4. **可読性**: コードの見通しが良くなる

```
[メイングラフ]
  ├─ create_plan
  ├─ execute_subtasks ←── サブグラフを呼び出す
  └─ create_answer

[サブグラフ]
  ├─ select_tools
  ├─ execute_tools
  └─ create_subtask_answer
```

</details>

---

### Think 3-3: Annotated + operator.add とは？

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

これにより：
- 追加分だけ返せばOK（全体を再構築する必要がない）
- 並列実行時も結果が正しくマージされる

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

SUBTASK_SYSTEM_PROMPT = """
あなたはXYZというシステムの質問応答のためにサブタスク実行を担当するエージェントです。
サブタスクはユーザーの質問に回答するために考えられた計画の一つです。

ツールの実行結果から得られた回答に必要なことは言語化してください。
回答できなかった場合は、その旨を言語化してください。
"""

CREATE_LAST_ANSWER_SYSTEM_PROMPT = """
あなたはXYZというシステムのヘルプデスク回答作成担当です。
サブタスクの結果をもとに回答を作成してください。

- 回答は質問者の意図を汲み取り、丁寧に作成してください
- 回答は簡潔で明確にすることを心がけてください
- 不確定な情報や推測を含めないでください
- 調べた結果から回答がわからなかった場合は、その旨を素直に回答に含めてください
"""

CREATE_LAST_ANSWER_USER_PROMPT = """
ユーザーの質問: {question}

サブタスクの結果:
{subtask_results}

回答を作成してください
"""
```

#### 3. `src/agent.py` を更新

Step 2から大幅に構造が変わります。主な変更点：

1. **新しいインポート**: `operator`, `Annotated`, `Sequence`, `Pregel`, `ChatCompletionMessageParam` など
2. **状態の拡張**: `AgentState` に `plan`, `current_step`, `subtask_results` を追加
3. **サブグラフの導入**: `SubGraphState` と `create_subgraph()` を新規追加
4. **新しいノード**: `create_plan`, `execute_subtasks`, `should_continue` を追加
5. **条件分岐**: `add_conditional_edges` でサブタスクをループ実行

```python
"""
Step 3: Plan追加
- Plan-and-Executeパターン
- サブグラフによる責務分離
- Annotated + operator.add による状態蓄積
"""

import json
import operator
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


# === ツールの準備 ===
TOOLS = [search_xyz_manual, search_xyz_qa]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# === 状態の定義 ===
class AgentState(TypedDict):
    """メイングラフの状態"""
    question: str
    plan: list[str]
    current_step: int
    # Annotated + operator.add で「追加分だけ返せばマージされる」
    subtask_results: Annotated[Sequence[Subtask], operator.add]
    answer: str


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
    """ツールを選択するノード"""

    print(f"  [SubGraph] select_tools: {state['subtask']}")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)
    openai_tools = [convert_to_openai_tool(tool) for tool in TOOLS]

    user_prompt = f"サブタスク: {state['subtask']}\n\n適切なツールを選択して実行してください。"
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SUBTASK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        tools=openai_tools,
        temperature=0,
    )

    if response.choices[0].message.tool_calls:
        ai_message: ChatCompletionMessageParam = {
            "role": "assistant",
            "tool_calls": [tc.model_dump() for tc in response.choices[0].message.tool_calls],
        }
        messages.append(ai_message)

    return {"messages": messages}


def execute_tools(state: SubGraphState) -> dict:
    """ツールを実行するノード"""

    print(f"  [SubGraph] execute_tools")

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

        tool_results.append(ToolResult(
            tool_name=tool_name,
            args=tool_args,
            results=result
        ))

        tool_message: ChatCompletionMessageParam = {
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False),
            "tool_call_id": tc["id"]
        }
        messages.append(tool_message)

    return {"messages": messages, "tool_results": tool_results}


def create_subtask_answer(state: SubGraphState) -> dict:
    """サブタスク回答を作成するノード"""

    print(f"  [SubGraph] create_subtask_answer")

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
    """サブグラフを実行するノード"""

    subtask = state["plan"][state["current_step"]]
    print(f"[Node] execute_subtask ({state['current_step'] + 1}/{len(state['plan'])}): {subtask}")

    subgraph = create_subgraph()

    result = subgraph.invoke({
        "question": state["question"],
        "plan": state["plan"],
        "subtask": subtask,
        "messages": [],
        "tool_results": [],
        "subtask_answer": "",
    })

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
    """最終回答を作成するノード"""

    print("[Node] create_answer")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # サブタスク結果を文字列に変換
    subtask_results_str = "\n\n".join([
        f"【{r.task_name}】\n{r.subtask_answer}"
        for r in state["subtask_results"]
    ])

    messages = [
        {"role": "system", "content": CREATE_LAST_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": CREATE_LAST_ANSWER_USER_PROMPT.format(
            question=state["question"],
            subtask_results=subtask_results_str,
        )},
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
    workflow.add_edge("create_plan", "execute_subtasks")

    # 条件分岐: サブタスクが残っていれば続行、なければ回答作成へ
    workflow.add_conditional_edges(
        "execute_subtasks",
        should_continue,
        {
            "continue": "execute_subtasks",  # ループ
            "finish": "create_answer"
        }
    )

    workflow.add_edge("create_answer", END)

    return workflow.compile()


# === 実行 ===
if __name__ == "__main__":
    app = create_graph()

    # 複数トピックの質問
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

### Think 3-4: ループの仕組み

> `add_conditional_edges` を使ったループの仕組みを図で描けますか？

<details>
<summary>回答</summary>

```
[START]
   ↓
[create_plan]
   ↓
[execute_subtasks] ←─────┐
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
`execute_subtasks` に戻ってループします。

</details>

---

### Think 3-5: Structured Outputの利点

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
- `add_conditional_edges` による条件分岐とループ
- サブグラフの作成（`StateGraph` → `compile()`）
- `Annotated` + `operator.add` による状態の蓄積

**次のステップ**: Reflectionを追加して、回答品質を自己評価する

---

## Step 4: Reflection追加

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

### Think 4-1: なぜ自己評価が必要？

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

### Think 4-2: Reflectionのトレードオフ

> Reflectionを入れることのデメリットは何でしょうか？

<details>
<summary>回答</summary>

1. **レイテンシの増加**: 評価のためのLLM呼び出しが追加される
2. **コストの増加**: API呼び出し回数が増える
3. **無限ループのリスク**: 適切な終了条件（MAX_CHALLENGE_COUNT）が必要

トレードオフを考慮して、本当に必要な場面でのみReflectionを使うことが重要。

</details>

---

### 実装

Step 3からの変更点を中心に説明します。

#### 1. `src/models.py` を更新

**変更点:**
- `ReflectionResult` クラスを追加（自己評価の結果を表現）
- `Subtask` クラスに `is_completed`, `challenge_count` フィールドを追加

```python
# === 追加 ===
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


# === 変更: Subtaskクラスにフィールド追加 ===
class Subtask(BaseModel):
    """サブタスクの実行結果"""
    task_name: str = Field(..., description="サブタスクの名前")
    tool_results: list[ToolResult] = Field(..., description="ツール実行結果")
    subtask_answer: str = Field(..., description="サブタスクの回答")
    is_completed: bool = Field(default=True, description="完了フラグ")      # 追加
    challenge_count: int = Field(default=1, description="試行回数")         # 追加
```

#### 2. `src/prompts.py` を更新

**変更点:** 2つのプロンプトを追加

```python
# === 追加 ===
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

**主な変更点:**
1. **インポート追加**: `ReflectionResult`, `SUBTASK_REFLECTION_USER_PROMPT`, `SUBTASK_RETRY_USER_PROMPT`
2. **定数追加**: `MAX_CHALLENGE_COUNT = 3`
3. **SubGraphState拡張**: `is_completed`, `challenge_count` を追加
4. **select_tools変更**: リトライ時は過去の対話履歴を使用
5. **create_subtask_answer変更**: messagesにassistantメッセージを追加
6. **新規関数追加**: `reflect_subtask`, `should_continue_subgraph`
7. **サブグラフ変更**: `reflect_subtask`ノードと条件分岐を追加

```python
# === 定数追加 ===
MAX_CHALLENGE_COUNT = 3


# === SubGraphState: フィールド追加 ===
class SubGraphState(TypedDict):
    """サブグラフの状態"""
    question: str
    plan: list[str]
    subtask: str
    is_completed: bool              # 追加
    messages: list[ChatCompletionMessageParam]
    challenge_count: int            # 追加
    tool_results: Annotated[Sequence[ToolResult], operator.add]
    subtask_answer: str


# === select_tools: リトライ対応 ===
def select_tools(state: SubGraphState) -> dict:
    """ツールを選択するノード"""
    # ...省略...

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

    # ...以下同様...


# === create_subtask_answer: messagesにassistant追加 ===
def create_subtask_answer(state: SubGraphState) -> dict:
    """サブタスク回答を作成するノード"""
    # ...省略...

    messages = list(state["messages"])
    messages.append({"role": "assistant", "content": subtask_answer})  # 追加

    return {"messages": messages, "subtask_answer": subtask_answer}


# === 新規追加: reflect_subtask ===
def reflect_subtask(state: SubGraphState) -> dict:
    """サブタスク回答を内省するノード"""

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    messages = list(state["messages"])
    messages.append({"role": "user", "content": SUBTASK_REFLECTION_USER_PROMPT})

    response = client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=messages,
        response_format=ReflectionResult,
        temperature=0,
    )

    reflection = response.choices[0].message.parsed
    if reflection is None:
        raise ValueError("Reflection result is None")

    messages.append({
        "role": "assistant",
        "content": reflection.model_dump_json()
    })

    update: dict = {
        "messages": messages,
        "challenge_count": state["challenge_count"] + 1,
        "is_completed": reflection.is_completed,
    }

    # 最大試行回数に達しても未完了の場合
    if update["challenge_count"] >= MAX_CHALLENGE_COUNT and not reflection.is_completed:
        update["subtask_answer"] = f"{state['subtask']}の回答が見つかりませんでした。"

    if reflection.is_completed:
        print(f"    ✓ 評価OK")
    else:
        print(f"    ✗ 評価NG: {reflection.advice}")

    return update


# === 新規追加: should_continue_subgraph ===
def should_continue_subgraph(state: SubGraphState) -> Literal["end", "continue"]:
    """サブグラフの継続判定"""
    if state["is_completed"] or state["challenge_count"] >= MAX_CHALLENGE_COUNT:
        return "end"
    else:
        return "continue"


# === create_subgraph: reflect_subtaskノードと条件分岐を追加 ===
def create_subgraph() -> Pregel:
    """サブグラフを作成する"""
    workflow = StateGraph(SubGraphState)

    workflow.add_node("select_tools", select_tools)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("create_subtask_answer", create_subtask_answer)
    workflow.add_node("reflect_subtask", reflect_subtask)  # 追加

    workflow.add_edge(START, "select_tools")
    workflow.add_edge("select_tools", "execute_tools")
    workflow.add_edge("execute_tools", "create_subtask_answer")
    workflow.add_edge("create_subtask_answer", "reflect_subtask")  # 変更

    # 追加: Reflectionの結果でループするか終了するか決定
    workflow.add_conditional_edges(
        "reflect_subtask",
        should_continue_subgraph,
        {"continue": "select_tools", "end": END}
    )

    return workflow.compile()


# === execute_subtasks: サブグラフ呼び出し時の初期状態を変更 ===
def execute_subtasks(state: AgentState) -> dict:
    """サブグラフを実行するノード"""
    # ...省略...

    result = subgraph.invoke({
        "question": state["question"],
        "plan": state["plan"],
        "subtask": subtask,
        "is_completed": False,      # 追加
        "challenge_count": 0,       # 追加
        "messages": [],
        "tool_results": [],
        "subtask_answer": "",
    })

    subtask_result = Subtask(
        task_name=subtask,
        tool_results=list(result["tool_results"]),
        subtask_answer=result["subtask_answer"],
        is_completed=result["is_completed"],        # 追加
        challenge_count=result["challenge_count"],  # 追加
    )

    return {"subtask_results": [subtask_result], "current_step": state["current_step"] + 1}
```

**実行部分の出力変更:**

```python
# === 実行部分: 出力フォーマット変更 ===
print("【サブタスク結果】")
for r in result["subtask_results"]:
    status = "✓" if r.is_completed else "✗"
    print(f"  {status} {r.task_name} (試行: {r.challenge_count}回)")
```

---

### 実行してみよう

```bash
uv run python -m src.agent
```

---

### Think 4-3: サブグラフ内のループ

> サブグラフ内の `should_continue_subgraph` はどのような条件で動作しますか？

<details>
<summary>回答</summary>

```
[select_tools] → [execute_tools] → [create_subtask_answer] → [reflect_subtask]
                                                                    ↓
                                                           {should_continue_subgraph}
                                                                    │
                                              ┌─────────────────────┴─────────────────────┐
                                              ↓                                           ↓
                                         "continue"                                     "end"
                                    (is_completed=False かつ                   (is_completed=True または
                                     challenge_count < MAX)                     challenge_count >= MAX)
                                              ↓                                           ↓
                                       [select_tools] ←─────────                        [END]
```

- **終了条件**: `is_completed=True`（評価OK）または `challenge_count >= MAX_CHALLENGE_COUNT`
- **継続条件**: `is_completed=False` かつ `challenge_count < MAX_CHALLENGE_COUNT`

</details>

---

### Step 4 まとめ

学んだこと：
- Reflectionパターンの実装
- サブグラフ内でのループ処理
- 会話履歴（messages）の管理
- 最大試行回数による終了条件

**次のステップ**: 並列実行でパフォーマンスを向上させる

---

## Step 5: 並列実行

### 目標
- LangGraphの`Send`機能を使って並列実行を実装する
- 複数サブタスクを同時に処理してパフォーマンスを向上させる

### 概念説明

Step 4までは直列実行：
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

### Think 5-1: なぜ並列実行が有効？

> 並列実行のメリットとデメリットを考えてみてください。

<details>
<summary>回答</summary>

**メリット:**
- 処理時間の短縮（複数タスクが同時に実行される）
- 効率的なリソース利用

**デメリット:**
- 実装が複雑になる
- デバッグが難しくなる
- タスク間で依存関係がある場合は使えない

今回のサブタスクは独立しているので、並列実行が有効です。

</details>

---

### Think 5-2: Sendの仕組み

> `Send` を返すと何が起きますか？

<details>
<summary>回答</summary>

`Send` は「このノードをこの状態で実行せよ」という指示です。

```python
Send("execute_subtasks", {"subtask": "タスク1", ...})
```

リストで複数の `Send` を返すと、それらが並列に実行されます。
各 `Send` は独立した状態を持ち、結果は `operator.add` でマージされます。

</details>

---

### 実装

Step 4からの変更点を中心に説明します。
**サブグラフ部分はStep 4と同じなので変更不要です。**

#### `src/agent.py` を更新

**主な変更点:**
1. **インポート追加**: `from langgraph.constants import Send`
2. **create_plan変更**: `current_step: 0` を返さない
3. **新規関数追加**: `route_subtasks`（Sendによる並列実行）
4. **execute_subtasks変更**: `current_step` 更新を削除
5. **should_continue削除**: 並列実行なのでループ不要
6. **create_graph変更**: `add_conditional_edges` と `set_finish_point` を使用

```python
# === インポート追加 ===
from langgraph.constants import Send  # 追加


# === create_plan: current_stepを返さない ===
def create_plan(state: AgentState) -> dict:
    """計画を作成するノード"""
    # ...省略...

    return {"plan": plan.subtasks}  # current_step: 0 を削除


# === 新規追加: route_subtasks ===
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
            }
        )
        for idx, _ in enumerate(state["plan"])
    ]


# === execute_subtasks: current_step更新を削除 ===
def execute_subtasks(state: AgentState) -> dict:
    """サブグラフを実行するノード"""

    subtask = state["plan"][state["current_step"]]
    print(f"[Node] execute_subtask: {subtask}")  # 番号表示を削除

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
        challenge_count=result["challenge_count"],
    )

    # current_stepの更新を削除（並列実行では不要）
    return {"subtask_results": [subtask_result]}


# === should_continue関数を削除 ===
# 並列実行ではメイングラフでのループが不要なため


# === create_graph: 並列実行対応に変更 ===
def create_graph() -> Pregel:
    """メイングラフを作成する"""
    workflow = StateGraph(AgentState)

    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_subtasks", execute_subtasks)
    workflow.add_node("create_answer", create_answer)

    workflow.add_edge(START, "create_plan")

    # 変更: 条件分岐でSendを返すと並列実行される
    workflow.add_conditional_edges(
        "create_plan",
        route_subtasks,
    )

    workflow.add_edge("execute_subtasks", "create_answer")

    # 変更: set_finish_pointを使用
    workflow.set_finish_point("create_answer")

    return workflow.compile()
```

**グラフ構造の変化:**

```
Step 4（直列実行）:
[create_plan] → [execute_subtasks] ←──┐
                      ↓              │
               {should_continue}     │
                      ├─ continue ───┘
                      ↓
                   finish
                      ↓
              [create_answer]

Step 5（並列実行）:
[create_plan]
      ↓
{route_subtasks} ─→ Send("execute_subtasks", step=0)
                 ─→ Send("execute_subtasks", step=1)
                 ─→ Send("execute_subtasks", step=2)
                          ↓ (並列実行後にマージ)
                   [create_answer]
```

---

### 実行してみよう

```bash
uv run python -m src.agent
```

---

### Think 5-3: 変更点の確認

> Step 4からStep 5への変更点を確認してください。何が変わりましたか？

<details>
<summary>回答</summary>

**主な変更点:**

1. **`route_subtasks` 関数の追加**
   - `Send` のリストを返す
   - 各サブタスクに対して並列実行を指示

2. **`add_conditional_edges` の変更**
   - `create_plan` → `route_subtasks` → 複数の `execute_subtasks`
   - ループ処理がなくなった（並列実行なので不要）

3. **`should_continue` の削除**
   - メイングラフでのループが不要になった

4. **`set_finish_point` の使用**
   - 複数の `execute_subtasks` が完了した後に `create_answer` に進む

</details>

---

### Think 5-4: chapter4との比較

> chapter4の実装と比較して、違いを確認してみてください。
> 構造は同じになっていますか？

<details>
<summary>確認ポイント</summary>

chapter4の `agent.py` と比較すると：
- メイングラフとサブグラフの分離 ✓
- `AgentState` と `SubGraphState` の分離 ✓
- `Send` による並列実行 ✓
- Reflection付きサブタスク実行 ✓

chapter4にはカスタムロガーやクラス化などの追加機能がありますが、
コアとなるエージェントの構造は同じです。

</details>

---

### Step 5 まとめ

学んだこと：
- `Send` による並列実行
- `add_conditional_edges` で `Send` のリストを返す
- `set_finish_point` の使い方
- 並列実行時の状態マージ

---

## 完成！

おめでとうございます！ 🎉

5つのステップを通じて、以下を学びました：

1. **Step 1**: LangGraphの基本（StateGraph, ノード, エッジ）
2. **Step 2**: Tool Callingの仕組み
3. **Step 3**: Plan-and-Executeパターン（サブグラフ、Annotated + operator.add）
4. **Step 4**: Reflectionによる自己評価とリトライ
5. **Step 5**: 並列実行（Send）

これで、chapter4と同等のAIエージェントを自分で構築できるようになりました！

### 次のチャレンジ

- プロンプトを改善して回答品質を向上させる
- 別のツール（例：Webスクレイピング、DB検索）を追加する
- ストリーミング対応を実装する
- エラーハンドリングを強化する
