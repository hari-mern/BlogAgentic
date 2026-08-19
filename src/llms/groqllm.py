from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

class GroqLLM:
    def __init__(self):
        load_dotenv()

    def get_llm(self):
        try:
            os.environ['GROQ_API_KEY']=self.groq_api_key=os.getenv("GROQ_API_KEY")
            model_name=os.getenv("GROQ_MODEL","qwen/qwen3.6-27b")
            llm_kwargs={"api_key":self.groq_api_key,"model_name":model_name}
            if "qwen" in model_name:
                llm_kwargs["reasoning_effort"]=os.getenv("GROQ_REASONING_EFFORT","none")
            llm=ChatGroq(**llm_kwargs)
            return llm
        except Exception as e:
            raise ValueError(f"Error occurred with exception : {e}")