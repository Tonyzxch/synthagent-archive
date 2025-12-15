import os
import json
import time
import random
import cv2
import numpy as np
import sys
import uuid
import argparse
from datetime import datetime
from playwright.sync_api import sync_playwright

from syn.gpt import GPTClient
from syn.args import APIProvider
from syn.tools import tools_ndarray_to_base64_image

from prompts import (
    prompt_select_deep_link, 
    prompt_evolve_task_description, 
    prompt_reverse_engineer_task
)

# 配置日志
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass

def get_page_screenshot_np(page):
    """获取截图并转为 numpy 格式"""
    try:
        screenshot_bytes = page.screenshot(timeout=5000)
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img_np
    except Exception as e:
        print(f"⚠️ 截图失败: {e}")
        return None

def save_screenshot_to_disk(img_np, output_dir):
    """保存截图到 outputs 目录"""
    if img_np is None: return None
    screenshot_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(screenshot_dir, filename)
    cv2.imwrite(filepath, img_np)
    return filepath

def get_interactive_elements(page, visited_texts=None):
    """获取页面交互元素"""
    if visited_texts is None: visited_texts = set()

    js_script = """
    () => {
        const elements = Array.from(document.querySelectorAll('a, button, [role="button"], h3, h4, .title, .job-name, .name'));
        return elements.map((el, index) => {
            const rect = el.getBoundingClientRect();
            const isVisible = rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.top <= window.innerHeight;
            let text = el.innerText.trim() || el.getAttribute('title') || el.getAttribute('aria-label') || "";
            text = text.replace(/\\n/g, ' ').substring(0, 100); 
            return { index: index, text: text, isVisible: isVisible, tagName: el.tagName };
        }).filter(item => item.isVisible && item.text.length > 1); 
    }
    """
    try:
        raw_elements = page.evaluate(js_script)
    except:
        return "", []
    
    blacklist = ["登录", "Login", "Sign", "Privacy", "Terms", "ICP", "网警"]
    filtered_elements = []
    seen_texts = set()

    for el in raw_elements:
        text = el['text']
        if text in visited_texts: continue
        if text in seen_texts: continue
        seen_texts.add(text)
        if any(bad in text for bad in blacklist): continue
        filtered_elements.append(el)

    random.shuffle(filtered_elements)
    selected_elements = filtered_elements[:60] # 限制数量防止 Context 溢出
    
    gpt_text_lines = []
    valid_indices = []
    for i, el in enumerate(selected_elements):
        gpt_text_lines.append(f"[{i}] {el['tagName']}: {el['text']}")
        valid_indices.append(el['index']) 
    
    return "\n".join(gpt_text_lines), valid_indices

def call_gpt_via_client(client, messages, model_name):
    try:
        # GPTClient.request 会自动处理重试和 Token 统计
        response = client.request(
            messages=messages,
            model=model_name,
            temperature=0.7,
            max_completion_tokens=1024,
            json_mode=True
        )
        return response.message.content
    except Exception as e:
        print(f"⚠️ GPT 调用失败: {e}")
        return None

def run_one_exploration(browser_context, start_url, max_depth, global_root_blacklist, gpt_client, model_name, output_dir):
    page = browser_context.new_page()
    try:
        print(f"🌍 访问: {start_url}")
        page.goto(start_url, timeout=60000, wait_until='domcontentloaded')
        page.wait_for_timeout(2000)

        current_task_desc = "Find a specific item or service."
        evolution_history = [] 
        history_breadcrumbs = ["Homepage"]
        current_depth = 0
        session_clicked_texts = set() 
        consecutive_scroll_count = 0

        while current_depth < max_depth:
            print(f"📍 [Depth {current_depth}] URL: {page.url}")
            screenshot_before = get_page_screenshot_np(page)
            
            # 黑名单逻辑
            current_blacklist = session_clicked_texts.union(global_root_blacklist) if current_depth == 0 else session_clicked_texts
            elements_text, valid_indices = get_interactive_elements(page, current_blacklist)

            if not elements_text:
                print("⚠️ 无交互元素，结束探索。")
                break

            # 1. 选择动作
            print("🤔 思考下一步...")
            messages = prompt_select_deep_link(
                url=page.url, 
                elements_text=elements_text, 
                screenshot=screenshot_before,
                current_depth=current_depth,
                max_depth=max_depth
            )
            # 使用框架 Client 调用
            content = call_gpt_via_client(gpt_client, messages, model_name)
            if not content: break

            try:
                choice = json.loads(content)
                chosen_id = choice.get("element_id")
            except:
                break
            
            # 滚动逻辑
            if chosen_id == -1:
                consecutive_scroll_count += 1
                if consecutive_scroll_count >= 3: break
                print("👇 向下滚动")
                page.evaluate("window.scrollBy(0, 800)")
                time.sleep(1)
                history_breadcrumbs.append("Scrolled")
                continue

            consecutive_scroll_count = 0

            # 点击逻辑
            if isinstance(chosen_id, int) and 0 <= chosen_id < len(valid_indices):
                original_index = valid_indices[chosen_id]
                raw_text = elements_text.split('\n')[chosen_id]
                clean_text = raw_text.split(':', 1)[-1].strip()
                
                session_clicked_texts.add(clean_text)
                if current_depth == 0: global_root_blacklist.add(clean_text)
                
                history_breadcrumbs.append(clean_text)
                print(f"👆 点击: {clean_text}")

                old_pages_count = len(browser_context.pages)
                try:
                    # 执行 JS 点击
                    page.evaluate(f"""
                        const els = Array.from(document.querySelectorAll('a, button, [role="button"], h3, h4, .title, .job-name, .name'));
                        if (els[{original_index}]) {{ els[{original_index}].click(); }}
                    """)
                    page.wait_for_timeout(3000)
                    
                    # 处理新标签页
                    if len(browser_context.pages) > old_pages_count:
                        page = browser_context.pages[-1]
                        try: page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except: pass
                    
                    screenshot_after = get_page_screenshot_np(page)

                    # 2. 意图演化
                    print(f"🔄 优化意图...")
                    evolve_msgs = prompt_evolve_task_description(
                        url=page.url, 
                        prev_task=current_task_desc, 
                        action_desc=f"Clicked on '{clean_text}'", 
                        screenshot_before=screenshot_before,
                        screenshot_after=screenshot_after
                    )
                    evolve_content = call_gpt_via_client(gpt_client, evolve_msgs, model_name)
                    
                    if evolve_content:
                        evolve_res = json.loads(evolve_content)
                        new_task = evolve_res.get("updated_task")
                        if new_task:
                            print(f"✨ 新意图: {new_task}")
                            current_task_desc = new_task
                            # 保存证据
                            img_path = save_screenshot_to_disk(screenshot_before, output_dir)
                            evolution_history.append({
                                "step": current_depth,
                                "action": clean_text,
                                "refined_task": new_task,
                                "screenshot": img_path
                            })

                    current_depth += 1
                    if "login" in page.url.lower(): break

                except Exception as e:
                    print(f"⚠️ 动作执行错误: {e}")
                    break
            else:
                break

        # 3. 生成最终任务
        print("🧠 生成最终任务...")
        final_screenshot = get_page_screenshot_np(page)
        if final_screenshot is None:
            final_screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        gen_messages = prompt_reverse_engineer_task(page.url, final_screenshot, history_breadcrumbs)
        content = call_gpt_via_client(gpt_client, gen_messages, model_name)
        if not content: return None

        result = json.loads(content)
        complex_task = result.get("complex_task") or current_task_desc

        print(f"✅ 任务生成: {complex_task}\n")
        
        # 返回符合 multi_exeagent 格式的数据
        return {
            "task": complex_task,
            "start_url": start_url,
            "sites": ["custom"], # 用于 config.target_env 设置
            "evolution_trace": evolution_history, 
            "final_url": page.url
        }

    except Exception as e:
        print(f"❌ 任务失败: {e}")
        return None
    finally:
        page.close()

def main():
    parser = argparse.ArgumentParser(description="Evolutionary Task Generator (Step 1)")
    parser.add_argument("--target_start_url", type=str, required=True, help="Start URL")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of tasks to generate")
    parser.add_argument("--max_depth", type=int, default=5, help="Exploration depth")
    parser.add_argument("--output_dir", type=str, default="outputs/synthagent/custom_evolution", help="Output directory")
    
    # API Params
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--openai_api_base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model", type=str, default="gpt-4o")
    
    args = parser.parse_args()

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%y-%m-%d-%H_%M_%S")
    final_output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 重定向日志
    sys.stdout = Logger(os.path.join(final_output_dir, "run.log"))
    tasks_file = os.path.join(final_output_dir, "tasks.jsonl")

    # 初始化 GPT 客户端
    print("🔧 初始化 GPT Client...")
    gpt_client = GPTClient(
        provider=APIProvider.openai,
        api_key=args.openai_api_key,
        base_url=args.openai_api_base
    )

    print(f"🚀 启动生成 | 目标: {args.target_start_url} | 数量: {args.num_seeds}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, # 后续大规模训练时改为 true
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox', '--disable-infobars']
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York"
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")

        global_root_blacklist = set()

        for i in range(args.num_seeds):
            print(f"\n🎬 === 任务 {i+1}/{args.num_seeds} ===")
            task_data = run_one_exploration(
                context, 
                args.target_start_url, 
                args.max_depth, 
                global_root_blacklist, 
                gpt_client, 
                args.model,
                final_output_dir
            )
            
            if task_data:
                # 实时写入 JSONL，防止程序崩溃丢失数据
                with open(tasks_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(task_data, ensure_ascii=False) + "\n")
                print(f"💾 已保存至 {tasks_file}")

        print(f"\n🎉 所有任务生成完毕！输出文件: {tasks_file}")
        
        # 打印 GPT Token 使用统计 (框架功能)
        print("\n📊 Token Usage:")
        print(gpt_client.token_usage)

if __name__ == "__main__":
    main()