from llama_index.core.agent import AgentRunner


class LLM:
    agent: AgentRunner

    def __init__(self, llm=None, tools=None, instruction="", tool_retriever=None):
        self.llm = llm
        self.tools = tools
        self.tool_retriever = tool_retriever
        self.instruction = instruction
        self.system_prompt = f"""You are a domain-specific assistant that is helpful, respectful and honest. Always 
                answer as helpfully as possible, while being safe. Your answers should not include any harmful, 
                unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are 
                socially unbiased and positive in nature. If a question does not make any sense, or is not factually 
                coherent, explain why instead of answering something not correct. If you don't know the answer to a 
                question, please don't share false information.

                You may be provided with tools to help you answer questions. Always ensure you use the result from 
                the most appropriate/relevant function tool to answer the original question in the user's prompt and comment
                on any provided data from the user or from the function result. Whenever you use a tool, explain your answer
                based on the tool result.

                Here is your domain-specific instruction:
                {self.instruction}
                """
