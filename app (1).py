import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


app = FastAPI(
    title="AI Coding Crew API",
    description="Developer, Tester and Manager workflow using LangGraph and Gemini"
)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)


class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


class TaskRequest(BaseModel):
    task: str


@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return output or error trace."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()

    except Exception:
        result = "Execution Error:\n" + traceback.format_exc()

    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a coding task."""

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios for this coding task:

{task_description}

Include:
- Standard cases
- Edge cases
- Expected behavior

Return the result as a numbered list.
"""

    response = llm.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)


def extract_content(content) -> str:
    """Safely extract text from Gemini response."""

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts)

    return str(content)


def developer_node(state: CrewState):
    task = state["messages"][-1].content

    prompt = f"""
Write a clean Python script to solve this task:

{task}

Return ONLY Python code.
Do not include explanations.
Do not include markdown.
"""

    response = llm.invoke(prompt)
    code = extract_content(response.content)

    return {"code": code}


def tester_node(state: CrewState):
    task = state["messages"][-1].content

    test_response = generate_test_cases.invoke({
        "task_description": task
    })

    test_cases = extract_content(test_response)

    execution_result = run_python_code.invoke({
        "code": state["code"]
    })

    report = f"""EXECUTION OUTPUT:
{execution_result}

TEST SCENARIOS:
{test_cases}
"""

    return {"report": report}


def manager_node(state: CrewState):
    return {"next_step": "completed"}


workflow = StateGraph(CrewState)

workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager")
workflow.add_edge("manager", END)

crew_app = workflow.compile()


@app.get("/")
def home():
    return {
        "message": "AI Coding Crew API is running successfully!"
    }


@app.post("/run")
def run_task(request: TaskRequest):

    initial_state = {
        "messages": [
            HumanMessage(content=request.task)
        ],
        "next_step": None,
        "code": None,
        "report": None
    }

    result = crew_app.invoke(
        initial_state,
        config={"recursion_limit": 50}
    )

    return {
        "task": request.task,
        "generated_code": result.get("code"),
        "report": result.get("report")
    }
