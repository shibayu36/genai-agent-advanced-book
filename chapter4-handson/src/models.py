from pydantic import BaseModel, Field


class Plan(BaseModel):
    """計画を表すモデル"""

    subtasks: list[str] = Field(..., description="質問に回答するために必要なサブタスクのリスト")


class ToolResult(BaseModel):
    """ツール実行結果"""

    tool_name: str = Field(..., description="ツールの名前")
    args: str = Field(..., description="ツールの引数")
    results: list[dict] = Field(..., description="ツールの結果")


class ReflectionResult(BaseModel):
    """リフレクションの結果"""

    advice: str = Field(..., description="評価がNGの場合のアドバイス。別のツールを試す、別の文言で検索するなど。")
    is_completed: bool = Field(..., description="サブタスクに対して正しく回答できているかの評価結果")


class Subtask(BaseModel):
    """サブタスクの実行結果"""

    task_name: str = Field(..., description="サブタスクの名前")
    tool_results: list[ToolResult] = Field(..., description="ツール実行結果")
    subtask_answer: str = Field(..., description="サブタスクの回答")
    is_completed: bool = Field(default=True, description="完了フラグ")
    challenge_count: int = Field(default=1, description="試行回数")
