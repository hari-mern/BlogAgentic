import re
from datetime import datetime
from src.states.blogstate import BlogState

def clean_response(content):
    """
    strip reasoning think tags so the response works with both
    thinking and non thinking models
    """
    if not isinstance(content, str):
        return content
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

class BlogNode:
    """
    A class to represent the blog node
    """

    def __init__(self,llm):
        self.llm=llm
    
    def title_creation(self,state:BlogState):
        """
        create the title for the blog
        """

        if "topic" in state and state["topic"]:
            prompt="""
            you are an expert blog content writer. use markdown formating.
            genreate a blog title for the {topic}. this title should be creative and seo friendly.
            use the current year {year} when needed.
            return only the title as a single markdown heading. do not include any explanation,
            bullet points, or extra text.
            """

        system_message=prompt.format(topic=state["topic"], year=datetime.now().year)
        response=self.llm.invoke(system_message)
        title=clean_response(response.content).lstrip("#").strip()
        return {"blog":{"title":title}}


    def content_generation(self,state:BlogState):
        """
        create the content for the blog
        """
        
        if "topic" in state and state["topic"]:
            system_prompt = """
            you are expert blog writer. use markdown formatting.
            generate a detailed blog content for the topic "{topic}" in the year {year}.
            start the blog with the markdown heading "# {title}" and then
            write the detailed content with a clear breakdown of sections.
            """

            system_message = system_prompt.format(topic = state["topic"], title = state["blog"]["title"], year = datetime.now().year)
            response = self.llm.invoke(system_message)
            return {"blog":{"title": state['blog']['title'], "content": clean_response(response.content)}}

