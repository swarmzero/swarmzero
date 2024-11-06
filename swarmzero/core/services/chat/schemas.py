from typing import List

from llama_index.core.llms import MessageRole
from pydantic import BaseModel


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatData(BaseModel):
    messages: List[Message]


class ChatHistorySchema(BaseModel):
    user_id: str
    session_id: str
    message: str
    role: str
    timestamp: str
