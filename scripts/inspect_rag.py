"""Print the actual compiled LangGraph without calling a model."""

from backend.docent import get_graph


if __name__ == "__main__":
    print(get_graph().get_graph().draw_mermaid())
