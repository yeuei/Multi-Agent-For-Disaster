from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import langgraph
import langgraph.graph
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, RemoveMessage, BaseMessage
from pydantic import BaseModel, Field
from typing import Literal


def get_llm(base_url = 'http://0.0.0.0:8500/v1', api_key = 'none', model_name = 'Qwen2.5-7B-Instruct', temperature = 0):
    print(f'正在使用{model_name}模型，base_url: {base_url}, api_key: {api_key}')
    return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                # max_tokens=16384,
                timeout=60,
                max_retries=2,
                base_url=base_url,
                api_key=api_key,
                # streaming=True
                )
class Route2Agent(BaseModel):
    question: str = Field(description="要发送给所选代理的具体问题")
    go_to: Literal['law_agent' , 'knowledge_agent' , 'websearch_agent', 'document_agent', 'emergency_agent', 'finish'] = Field(description="要调用的代理名称或者选择finish")

class Summarize_History(BaseModel):
    information: str = Field(
        description="在充分理解整体任务目标和全部对话上下文的基础上，用清晰的方式进行总结，给出目前为止尽可能回答整体任务的综合回答。"
    )
    is_ok: Literal['yes','no'] = Field(
        description="如果当前给出的总结性回答已经解决了整体任务目标，则为'yes'；如果仍需要进一步信息或补充回答，则为 'no'。"
    )

class StructureAgent():
    def __init__(self, llm, prompt, StrutureClass:BaseModel) -> None:
        self.sys_prompt = prompt
        self.StrutureClass = StrutureClass
        self.llm_struture = llm.with_structured_output(self.StrutureClass)
        del prompt
        del StrutureClass
    def __name__(self) -> str:
        return f"prompt:{self.sys_prompt}, Structure:{self.StrutureClass}"
def draw_flow(graph, save_path = None):
    try:
        # 使用 Mermaid 生成图表并保存为文件
        mermaid_code = graph.get_graph().draw_mermaid_png()
        if save_path is None:
            save_path = 'graph.jpg'
        with open(save_path, "wb") as f:
            f.write(mermaid_code)

        # 使用 matplotlib 显示图像
        img = mpimg.imread(save_path)
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        if(save_path is not None):
            plt.savefig(save_path)
        plt.show()
    except Exception as e:
        print(f"报错： {e}")
def draw_ascii(graph, save_path = None):
    graph.get_graph().print_ascii()