from typing import Optional
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

class SummarizerTool:
    """Tool for summarizing text"""
    
    def __init__(self, model_name: str = "gpt-4", max_length: int = 500):
        self.llm = ChatOpenAI(model_name=model_name)
        self.max_length = max_length
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """Create a LangChain chain for summarization"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes text concisely."),
            ("user", "Please summarize the following text in {max_length} words or less:\n\n{text}")
        ])
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt
        )
    
    def create_tool(self) -> Tool:
        """Create a LangChain tool for summarization"""
        return Tool(
            name="summarize",
            description="Summarize a piece of text",
            func=self._summarize
        )
    
    def _summarize(self, text: str) -> str:
        """Summarize the given text"""
        try:
            result = self.chain.run(
                text=text,
                max_length=self.max_length
            )
            return result
        except Exception as e:
            return f"Error summarizing text: {str(e)}" 