# python sdk
import asyncio
from langgraph_sdk import get_client
from typing import Dict, Any, List, Optional
import json
from langchain_core.messages import HumanMessage
import re
import ast
import traceback
import os

# def _normalize_quotes(s: str) -> str:
#     return (
#         s.replace('“', '"').replace('”', '"')
#          .replace('‘', "'").replace('’', "'")
#     )
# def _parse_list_string(s: str):
#     s = s.strip()
#     s = _normalize_quotes(s)

#     # 1) 首选 ast（兼容单引号）
#     try:
#         val = ast.literal_eval(s)
#         if isinstance(val, list):
#             return val
#     except Exception:
#         pass

#     # 2) 次选 JSON（需要双引号）
#     try:
#         val = json.loads(s)
#         if isinstance(val, list):
#             return val
#     except Exception:
#         pass

#     # 3) 兜底：形如 A, B 或 'A','B' 没有方括号
#     if not (s.startswith('[') and s.endswith(']')) and ',' in s:
#         parts = [p.strip().strip('"').strip("'") for p in s.split(',')]
#         return parts

#     # 仍失败则返回 None 表示没解析成 list
#     return None

def extract_and_listify(text):
    """
    提取 \\box{内容} 并转为列表，支持不规范的无引号列表（如 [A, B]）。
    """
    if type(text) != str:
        return None, False
    pattern = r'\\box\{(.*?)\}'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        content_str = match.group(1).strip()
        
        # --- 尝试 1: 标准 Python 格式解析 (最安全/准确) ---
        try:
            parsed_content = ast.literal_eval(content_str)
            if isinstance(parsed_content, list):
                return parsed_content, True
            else:
                return content_str, False
                
        except (ValueError, SyntaxError):
            # --- 尝试 2: 容错处理 (针对 [A, B] 这种无引号情况) ---
            # 判断是否像是一个列表（以 [ 开头，以 ] 结尾）
            if content_str.startswith('[') and content_str.endswith(']'):
                # 1. 去掉首尾的 [ 和 ]
                inner_str = content_str[1:-1]
                
                # 2. 如果是空列表 []
                if not inner_str.strip():
                    return [], True
                
                # 3. 按逗号分割，并清理每一项的空白和可能存在的引号
                # 比如：[A,  'B', "C"] -> 都能处理
                raw_list = inner_str.split(',')
                final_list = []
                for item in raw_list:
                    # 去掉两端空白
                    item = item.strip()
                    # 去掉可能存在的单引号或双引号 (兼容混合写法)
                    # 比如用户写了 [A, 'B']，如果不strip引号，'B' 会变成 "'B'"
                    item = item.strip("'\"") 
                    final_list.append(item)
                
                return final_list, True
            
            # 如果连方括号都没有，或者格式彻底乱了，返回失败
            return content_str, False
    else:
        return None, False

    

async def main(question:str = '你好'):
    # 1. Connect to the LangGraph deployed project
    # client = get_client(url="http://localhost:2024")
    client = get_client(url="http://localhost:2024",api_key="langsmith_api_key", headers={
        "Content-Type": "application/json",
        "X-Api-Key": "langsmith_api_key"
    })
    
    # 2. Create a thread
    thread = await client.threads.create()
    print(f"Thread created successfully, Thread ID: {thread['thread_id']}")
    
    # 3. Prepare the input message
    user_message = question
    
    print(f"User input: {user_message}")
    print("=" * 50)
    print("Starting stream output:")
    print("=" * 50)

    final_ans = ''
    
    # 4. Stream the output
    last_data = None
    got_valid_output = False
    try:
        # Note: It's best practice to pass the thread_id to continue the conversation in the same thread.
        async for chunk in client.runs.stream(
            thread_id=thread['thread_id'],
            assistant_id="99776832-b1d6-4543-9bf9-8094ae4a8a3f",
            input={
                # Note: Use the variable `user_message`, not the string "user_message".
                "pre_messages": [HumanMessage(content=user_message)]
            },
            stream_mode=["values"],
        ):
            # Print each streaming data chunk
            if chunk:
                print(f"Received new event, type is: {chunk.event}...")
                print(chunk.data)
                print("\n\n")
                # 记录最后一次流的数据
                last_data = chunk.data
                # 当数据包含预期输出（例如 pre_messages 或 values）时，认为有效
                if isinstance(chunk.data, dict) and (
                    "pre_messages" in chunk.data or "values" in chunk.data or "output" in chunk.data
                ):
                    got_valid_output = True
    except Exception as e:
        print(f"An error occurred during streaming output: {e}")
        traceback.print_exc()
        return e, False
    
    # 根据是否拿到有效输出决定 is_ok
    if got_valid_output and last_data is not None:
        return last_data, True
    else:
        return {"error": "no valid output from stream", "last_data": last_data}, False

# Run the main function
if __name__ == "__main__":
    data_path = '输入data_path'
    data_name = {'law': '法律法规数据集.json', 'domain':'领域知识数据集.json', 'emergency':'应急科学数据集.json', 'sql':'灾害数据数据集.json', 'DisasterQA':'DisasterQA.json'}
    # 错误1: agent 没有正确运行
    # 错误2: agent 没有按照正确格式回答
    # 错误3: agent 回答错误
    for i in data_name.keys():
        json_file = data_path + '/' + data_name[i]
        print(f"Processing {i}: {json_file}")
        acc = 0
        total = 0
        with open(json_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        # === 进度恢复设置 ===
        progress_dir = '/home/25-fengyuan/Disaster_Agent数据集收集/test_details'
        os.makedirs(progress_dir, exist_ok=True)
        progress_file = f"{progress_dir}/{data_name[i].split('.')[0]}_progress.txt"
        last_done = 0
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as pf:
                    last_done = int((pf.read() or '0').strip() or '0')
            except Exception:
                last_done = 0

        for idx, item in enumerate(test_data):
            # 跳过已处理的数据
            if idx < last_done:
                continue
            erros = ''
            is_right = False
            # 获取问题
            question = item.get('问题', '')
            option = item.get('选项', {})
            figure = item.get('标签', {}).get('题目类型', '')
            right_ans = item.get('正确答案', [])
            question = f'题目类型:{figure}\n问题:\n{question}\n选项: {option}'
            # 记录结果
            result, is_ok = asyncio.run(main(question=question))

            if is_ok:
                print(f"Result type: {type(result)}")
                final_ans = result.get('pre_messages', [{"content": ['错误']}])[-1].get('content', '')
                # 记录是否正确
                total += 1
                option,is_list= extract_and_listify(final_ans)
                if is_list:
                    if set(option) == set(right_ans):
                        acc += 1
                        is_right = True
                    else:
                        erros += '错误3: agent 回答错误\n'
                else:
                    erros += '错误2: agent 没有按照正确格式回答\n'
            else:
                erros += '错误1: agent 没有正确运行\n'
            with open(f'/home/25-fengyuan/Disaster_Agent数据集收集/test_details/{data_name[i].split('.')[0] + ".jsonl"}', 'a', encoding='utf-8') as f:
                item['错误'] = erros
                item['是否正确'] = is_right
                item['history'] = result
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                
                print('========')
                print(f'final_ans: {option}')
                print(f'right_ans: {right_ans}')
                print(f'acc:{acc}/{total}')
                print(f'pro:{total}/{len(test_data)}')
                print('========')
            # 更新进度
            try:
                with open(progress_file, 'w', encoding='utf-8') as pf:
                    pf.write(str(idx + 1))
            except Exception:
                pass
        with open(f'/home/25-fengyuan/Disaster_Agent数据集收集/test_details/{data_name[i].split('.')[0] + ".jsonl"}', 'a', encoding='utf-8') as f:
            f.write(f'准确率: {acc}/{total}')
        # 数据集完成后将进度标记为完成（可选）
        try:
            with open(progress_file, 'w', encoding='utf-8') as pf:
                pf.write(str(len(test_data)))
        except Exception:
            pass
    