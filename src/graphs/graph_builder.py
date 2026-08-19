from langgraph.graph import END, START, StateGraph

from src.llms.groqllm import GroqLLM
from src.nodes.blog_node import BlogNode
from src.states.blogstate import BlogState


class GraphBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.blog_node_obj = BlogNode(llm)

    def build_topic_graph(self):
        """Graph that generates a blog from a topic."""
        graph = StateGraph(BlogState)

        graph.add_node("title_creation", self.blog_node_obj.title_creation)
        graph.add_node("content_generator", self.blog_node_obj.content_generator)

        graph.add_edge(START, "title_creation")
        graph.add_edge("title_creation", "content_generator")
        graph.add_edge("content_generator", END)

        return graph

    def build_language_graph(self):
        """Graph that generates a blog from a topic and translates it to a language."""
        graph = StateGraph(BlogState)

        graph.add_node("title_creation", self.blog_node_obj.title_creation)
        graph.add_node("content_generator", self.blog_node_obj.content_generator)
        graph.add_node(
            "hindi_translation",
            lambda state: self.blog_node_obj.translation({**state, "current_language": "hindi"}),
        )
        graph.add_node(
            "french_translation",
            lambda state: self.blog_node_obj.translation({**state, "current_language": "french"}),
        )
        graph.add_node("route", self.blog_node_obj.route)

        graph.add_edge(START, "title_creation")
        graph.add_edge("title_creation", "content_generator")
        graph.add_edge("content_generator", "route")

        graph.add_conditional_edges(
            "route",
            self.blog_node_obj.route_decision,
            {"hindi": "hindi_translation", "french": "french_translation"},
        )

        graph.add_edge("hindi_translation", END)
        graph.add_edge("french_translation", END)

        return graph

    def setup_graph(self, usecase):
        if usecase == "topic":
            return self.build_topic_graph().compile()
        if usecase == "language":
            return self.build_language_graph().compile()
        raise ValueError(f"Unknown usecase '{usecase}'. Supported: 'topic', 'language'.")


# Instantiated at import time so LangGraph Studio can load the compiled graph.
llm = GroqLLM().get_llm()
graph_builder = GraphBuilder(llm)
graph = graph_builder.build_language_graph().compile()