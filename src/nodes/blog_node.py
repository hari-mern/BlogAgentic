from src.states.blogstate import BlogState

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
            """

        system_message=prompt.format(topic=state["topic"])
        response=self.llm.invoke(system_message)
        return {"blog":{"title":response.content}}


    def content_generation(self,state:BlogState):
        """
        create the content for the blog
        """
        
        if "topic" in state and state["topic"]:
            system_prompt = """
            you are expert blog writer. use markdown formatting.
            generate a detailed blog content with detailed breakdown for the {topic}
            """

            system_message = system_prompt.format(topic = state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog":{"title": state['blog']['title'], "content": response.content}}

