from typing import Optional, Dict, Any
from langchain.tools import Tool
from memory.ollama_llm import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from memory.error_memory import ErrorMemory

class SummarizerTool:
    """Tool for summarizing text"""
    
    def __init__(self, model_name: str = "llama2", max_length: int = 500, error_memory: Optional[ErrorMemory] = None):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.3,  # Lower temperature for more focused summaries
            max_tokens=max_length
        )
        self.max_length = max_length
        self.chain = self._create_chain()
        self.error_memory = error_memory
    
    def _create_chain(self) -> RunnableSequence:
        """Create a LangChain chain for summarization using RunnableSequence"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes text concisely."),
            ("user", "Please summarize the following text in {max_length} words or less:\n\n{text}")
        ])
        
        return prompt | self.llm
    
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
            result = self.chain.invoke({
                "text": text,
                "max_length": self.max_length
            })
            return result.content
        except Exception as e:
            error_msg = f"Error summarizing text: {str(e)}"
            
            # Log to error memory if available
            if self.error_memory:
                self.error_memory.add_error(
                    error_msg,
                    context={
                        "tool": "summarizer",
                        "error_type": type(e).__name__,
                        "text_length": len(text),
                        "model": self.llm.model
                    }
                )
            
            return error_msg 