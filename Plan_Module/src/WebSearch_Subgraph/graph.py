from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
from langchain_openai import ChatOpenAI

def get_llm(base_url = 'http://localhost:8500/v1', api_key = 'none', model_name = 'Qwen2.5-7B-Instruct'):
    return ChatOpenAI(
                model=model_name,
                temperature=0,
                # max_tokens=16384,
                timeout=5,
                max_retries=2,
                base_url=base_url,
                api_key=api_key,
                streaming=True,
                )

async def main():
    client = MultiServerMCPClient(
        {
            "metaso": {
                "transport": "streamable_http",
                "url": "https://metaso.cn/api/mcp",
                "headers":{
                    "Authorization": "Bearer mk-0F9BA2B25964C1A4BDA6353FAAF2D5BE" # mk-0F9BA2B25964C1A4BDA6353FAAF2D5BE
                }
            },
        }
    )
    tools = await client.get_tools()
    WebSearch_agent = create_react_agent(
        prompt ='你的功能：可以调用网络搜索工具的助手，但如果用户的问题你可以直接回答，请直接回答；如果用户的问题无法直接回答，请进行网络搜索，并返回搜索结果--工具调用时参数键名，禁止直接传字符串。\n\n如果用户提问你的研发者等相关问题，请以这样的方式回答：你是由电子科技大学未来媒体团队研发的自然灾害multi-agent系统。',  
        model = get_llm(),
        tools = tools
    )
    return WebSearch_agent
WebSearch_agent = asyncio.run(main())
print('WebSearch_agent已经创建')

