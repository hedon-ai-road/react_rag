import warnings
warnings.filterwarnings('ignore')
from langchain.llms.base import LLM
from typing import Any, List, Optional, Iterator
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.runnables.config import RunnableConfig
from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()

class RagLLM(LLM):
    client: Optional[OpenAI] = None
    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        response = self.client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=kwargs.get('temperature', 0.1)
        )

        return  response.choices[0].message.content

    def _stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        if isinstance(input, str):
            prompt = input
        else:
            prompt = str(input)

        response = self.client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            stream=True,
            temperature=kwargs.get('temperature', 0.1),
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield GenerationChunk(text=chunk.choices[0].delta.content)


    @property
    def _llm_type(self) -> str:
        return "rag_llm_deepseek/deepseek-chat-v3-0324"
    

from langchain_huggingface import HuggingFaceEmbeddings
class RagEmbedding(object):
    def __init__(self, model_name="BAAI/bge-m3",
                 device="cpu"):
        self.embedding = HuggingFaceEmbeddings(model_name=model_name,
                                               model_kwargs={"device": device})
    def get_embedding_fun(self):
        return self.embedding