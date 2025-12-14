import os
import json
import time
import random
import cv2
import numpy as np
import sys
import uuid
from datetime import datetime
from playwright.sync_api import sync_playwright
from openai import OpenAI

# 导入所有需要的 Prompt 函数
from prompts import (
    prompt_select_deep_link, 
    prompt_evolve_task_description, 
    prompt_reverse_engineer_task
)
# 导入工具函数 (用于处理 base64 图片)
from syn.tools import tools_ndarray_to_base64_image

# 脚本参数配置
# START_URL = "https://bj.lianjia.com/"  # 目标网站
# START_URL = "http://18.216.88.140:9980/"  # CLASSIFIEDS
START_URL = "http://18.216.88.140:7770"  # SHOPPING
# START_URL = "http://18.216.88.140:9999"  # REDDIT
NUM_SEEDS = 10                         # 生成多少个任务
MAX_DEPTH = 5                          # 探索深度
HEADLESS = False                       # 是否无头模式

# API 配置
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-8ii5hArDqmElR01FWP0ZGEFrGNaXIdaDHyalDtdl6ohFS3ie")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://xiaoai.plus/v1")
MODEL_NAME = "gpt-4o"

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# 全局日志记录类
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

# 创建本次运行的唯一目录
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("outputs", f"{RUN_TIMESTAMP}")
SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")
TASKS_FILE = os.path.join(OUTPUT_DIR, "tasks.jsonl")
LOG_FILE = os.path.join(OUTPUT_DIR, "run.log")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
sys.stdout = Logger(LOG_FILE)

def get_page_screenshot_np(page):
    """获取截图并转为 numpy 格式"""
    try:
        screenshot_bytes = page.screenshot(timeout=5000)
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None: return None
        return img_np
    except Exception as e:
        print(f"⚠️ 截图失败: {e}")
        return None

def save_screenshot_to_disk(img_np):
    """保存截图到硬盘，返回相对路径"""
    if img_np is None: return None
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(filepath, img_np)
    return filepath

def get_interactive_elements(page, visited_texts=None):
    """获取页面可交互元素，并过滤掉已点击过的文本"""
    if visited_texts is None:
        visited_texts = set()

    js_script = """
    () => {
        // 抓取范围尽可能广：链接、按钮、标题、图片容器
        const elements = Array.from(document.querySelectorAll('a, button, [role="button"], h3, h4, .title, .job-name, .name'));
        return elements.map((el, index) => {
            const rect = el.getBoundingClientRect();
            const isVisible = rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.top <= window.innerHeight;
            
            // 获取文本，优先取 title 或 aria-label
            let text = el.innerText.trim() || el.getAttribute('title') || el.getAttribute('aria-label') || "";
            text = text.replace(/\\n/g, ' ').substring(0, 100); 
            
            return {
                index: index,
                text: text,
                isVisible: isVisible,
                tagName: el.tagName
            };
        }).filter(item => item.isVisible && item.text.length > 1); 
    }
    """
    try:
        raw_elements = page.evaluate(js_script)
    except:
        return "", []
    
    # 黑名单过滤
    blacklist = [
        "登录", "注册", "Login", "Sign", "隐私", "Privacy", "条款", "Terms", 
        "APP", "下载", "Download", "关于", "About", "广告", "Ads", "ICP", "网警"
    ]
    
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
    selected_elements = filtered_elements[:60]
    
    gpt_text_lines = []
    valid_indices = []

    for i, el in enumerate(selected_elements):
        gpt_text_lines.append(f"[{i}] {el['tagName']}: {el['text']}")
        valid_indices.append(el['index']) 
    
    return "\n".join(gpt_text_lines), valid_indices

def call_gpt_4o_universal(messages):
    """封装 GPT 调用，自动处理 Base64 图片前缀"""
    final_messages = []
    for msg in messages:
        new_content = []
        for item in msg['content']:
            if item['type'] == 'image_url':
                original_url = item['image_url']['url']
                # OpenAI 原生 API 需要 data:image 前缀
                if not original_url.startswith("data:"):
                    new_url = f"data:image/jpeg;base64,{original_url}"
                else:
                    new_url = original_url
                
                new_content.append({
                    "type": "image_url",
                    "image_url": {"url": new_url}
                })
            else:
                new_content.append(item)
        final_messages.append({"role": msg["role"], "content": new_content})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=final_messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"⚠️ OpenAI API 调用失败: {e}")
        return None

def run_one_exploration(browser_context, global_root_blacklist):
    page = browser_context.new_page()
    try:
        print(f"🌍 访问: {START_URL}")
        page.goto(START_URL, timeout=60000, wait_until='domcontentloaded')
        page.wait_for_timeout(3000)

        current_task_desc = "Find a specific item or service."
        evolution_history = [] 
        history_breadcrumbs = ["Homepage"]
        current_depth = 0
        session_clicked_texts = set() 
        consecutive_scroll_count = 0

        while current_depth < MAX_DEPTH:
            print(f"📍 深度 {current_depth}: {page.url}")
            
            screenshot_before = get_page_screenshot_np(page)

            # 如果当前是首页 (Depth 0)，我们把"这一轮点过的" 和 "之前几轮点过的" 都过滤掉
            if current_depth == 0:
                # 合并当前会话的黑名单 + 全局的首页黑名单
                current_blacklist = session_clicked_texts.union(global_root_blacklist)
            else:
                # 如果不是首页，只过滤当前会话点过的即可（允许不同任务在子页面重复点击相同的筛选）
                current_blacklist = session_clicked_texts
                
            elements_text, valid_indices = get_interactive_elements(page, current_blacklist)

            if not elements_text:
                print("⚠️ 无可交互元素，停止。")
                break

            # 选动作
            print("🤔 GPT 正在挑选动作...")
            messages = prompt_select_deep_link(
                url=page.url, 
                elements_text=elements_text, 
                screenshot=screenshot_before,
                current_depth=current_depth,
                max_depth=MAX_DEPTH
            )
            content = call_gpt_4o_universal(messages)
            if not content: break

            try:
                choice = json.loads(content)
                chosen_id = choice.get("element_id")
            except:
                print("⚠️ JSON解析失败")
                break
            
            # 处理滚动动作
            if chosen_id == -1:
                consecutive_scroll_count += 1

                if consecutive_scroll_count >= 3:
                    print("⚠️ 连续滚动过多，停止探索。")
                    break
                
                print("👇 GPT 决定：当前屏幕没意思，向下滑动...")
                page.evaluate("window.scrollBy(0, 800)") # 向下滚 800 像素
                time.sleep(1) # 等滚动动画
                
                history_breadcrumbs.append("Scrolled Down")
                continue # 继续下一轮循环

            consecutive_scroll_count = 0  # 重置连续滚动计数器

            if isinstance(chosen_id, int) and 0 <= chosen_id < len(valid_indices):
                original_index = valid_indices[chosen_id]
                raw_text = elements_text.split('\n')[chosen_id]
                # 清洗文本，去掉 [0] A: 前缀
                clean_text = raw_text.split(':', 1)[-1].strip()

                # 记录到历史，防止下一轮再点
                session_clicked_texts.add(clean_text)

                # 如果是首页的操作，记录到全局黑名单
                if current_depth == 0:
                    global_root_blacklist.add(clean_text)
                
                history_breadcrumbs.append(clean_text)
                print(f"👆 点击: {clean_text}")

                # 记录点击前的页面数
                old_pages_count = len(browser_context.pages)

                # 执行点击
                try:
                    page.evaluate(f"""
                        const els = Array.from(document.querySelectorAll('a, button, [role="button"], h3, h4, .title, .job-name, .name'));
                        if (els[{original_index}]) {{ els[{original_index}].click(); }}
                    """)
                    
                    page.wait_for_timeout(3000) # 等待动作生效
                    
                    # 检查新标签页
                    new_pages = browser_context.pages
                    if len(new_pages) > old_pages_count:
                        print(f"🔀 【视角切换】新标签页弹出 (Total: {len(new_pages)})")
                        new_page = new_pages[-1]
                        try:
                            new_page.wait_for_load_state("domcontentloaded", timeout=10000)
                        except:
                            print("⚠️ 新页面加载超时")
                        page = new_page 
                    else:
                        print("🔄 原页跳转或DOM变化...")
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except:
                            pass
                    
                    print(f"🔗 当前 URL: {page.url}")

                    screenshot_after = get_page_screenshot_np(page)

                    # 优化意图
                    print(f"🔄 正在结合前后视图优化意图...")
                    evolve_msgs = prompt_evolve_task_description(
                        url=page.url, 
                        prev_task=current_task_desc, 
                        action_desc=f"Clicked on '{clean_text}'", 
                        screenshot_before=screenshot_before, # 前图
                        screenshot_after=screenshot_after    # 后图
                    )
                    evolve_content = call_gpt_4o_universal(evolve_msgs)
                    
                    if evolve_content:
                        try:
                            evolve_res = json.loads(evolve_content)
                            new_task = evolve_res.get("updated_task")
                            if new_task:
                                print(f"✨ 意图进化: {new_task}")
                                current_task_desc = new_task
                                
                                # 只保存前图作为训练数据
                                img_path = save_screenshot_to_disk(screenshot_before)

                                evolution_history.append({
                                    "step": current_depth + 1,
                                    "action": clean_text,
                                    "refined_task": new_task,
                                    "screenshot": img_path # 记录图片路径
                                })
                        except Exception as e:
                            print(f"⚠️ 意图解析失败: {e}")

                    current_depth += 1
                    
                    url_lower = page.url.lower()
                    if "login" in url_lower or "passport" in url_lower:
                        print("🛑 撞到了登录页，停止。")
                        break

                except Exception as e:
                    print(f"⚠️ 点击或切换失败: {e}")
                    break
            else:
                print("⚠️ GPT 选了无效ID，跳过。")
                break

        # 生成最终任务
        print("🧠 正在生成最终任务指令...")

        final_screenshot = get_page_screenshot_np(page)
        
        # 如果截图失败，就创建一个空的黑色图片，防止报错
        if final_screenshot is None:
            print("⚠️ 最终截图失败，使用空白图片代替...")
            final_screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 保存最终页面截图
        final_img_path = save_screenshot_to_disk(final_screenshot)

        gen_messages = prompt_reverse_engineer_task(page.url, final_screenshot, history_breadcrumbs)
        content = call_gpt_4o_universal(gen_messages)
        if not content: return None

        result = json.loads(content)
        complex_task = result.get("complex_task") or current_task_desc

        print(f"✅ 最终任务: {complex_task}\n")
        
        return {
            "task": complex_task,
            "start_url": START_URL,
            "path": history_breadcrumbs,
            "evolution_trace": evolution_history,
            "final_screenshot": final_img_path
        }

    except Exception as e:
        print(f"❌ 错误: {e}")
        return None
    finally:
        page.close()

def main():
    # 打印运行信息
    print(f"🚀 任务开始！输出目录: {OUTPUT_DIR}")
    
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        if k in os.environ: del os.environ[k]

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox', '--disable-infobars']
        )
        
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            locale="en-US", # 改成英文环境适合 Shopping 网站
            timezone_id="America/New_York"
        )
        
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

        global_root_blacklist = set()
        
        for i in range(NUM_SEEDS):
            print(f"🚀 === 任务 {i+1}/{NUM_SEEDS} ===")
            try:
                task_data = run_one_exploration(context, global_root_blacklist)
                if task_data:
                    with open(TASKS_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(task_data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"❌ 本次任务异常: {e}")

if __name__ == "__main__":
    main()