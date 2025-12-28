from pydantic import BaseModel, Field


class Plan(BaseModel):
    """計画を表すモデル"""

    subtasks: list[str] = Field(..., description="質問に回答するために必要なサブタスクのリスト")


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
