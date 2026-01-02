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
    prompt_reverse_engineer_task,
    prompt_audit_trajectory
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
        screenshot_bytes = page.screenshot(timeout=10000)
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
    if visited_texts is None: visited_texts = set()

    # JS 脚本：增加计算元素到中心的距离，优先保留主要内容区的元素
    js_script = """
    () => {
        const selectors = 'a, button, [role="button"], h3, h4, .title, .job-name, .name, input, textarea, select';
        const elements = Array.from(document.querySelectorAll(selectors));
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const centerX = viewportWidth / 2;
        const centerY = viewportHeight / 2;

        return elements.map((el, index) => {
            const rect = el.getBoundingClientRect();
            const isVisible = rect.width > 0 && rect.height > 0 && 
                              rect.top >= 0 && rect.top <= viewportHeight &&
                              rect.left >= 0 && rect.left <= viewportWidth;
            
            // 计算距离中心的距离，用于排序
            const dist = Math.sqrt(Math.pow(rect.left + rect.width/2 - centerX, 2) + Math.pow(rect.top + rect.height/2 - centerY, 2));
            
            // 优先取 value (针对有默认值的 input)，然后取 innerText 等
            let text = el.value || el.innerText.trim() || el.getAttribute('title') || el.getAttribute('aria-label') || el.getAttribute('placeholder') || "";
            let tagName = el.tagName;
            
            // 简单清洗
            text = text.replace(/\\n/g, ' ').substring(0, 80);
            
            return { index: index, text: text, isVisible: isVisible, tagName: tagName, dist: dist };
        }).filter(item => item.isVisible && (item.text.length > 1 || item.tagName === 'INPUT' || item.tagName === 'SELECT')); 
    }
    """
    try:
        raw_elements = page.evaluate(js_script)
    except:
        return "", []
    
    blacklist = ["登录", "Login", "Sign", "Privacy", "Terms", "ICP", "公安", "Copyright"]
    filtered_elements = []
    seen_texts = set()

    # 排序策略：1. 输入框最优先 2. 距离中心越近越优先
    raw_elements.sort(key=lambda x: (x['tagName'] != 'INPUT', x['dist'])) 

    for el in raw_elements:
        text = el['text']
        if text in visited_texts: continue
        if text in seen_texts: continue
        # 严格去重
        seen_texts.add(text) 
        if any(bad in text for bad in blacklist): continue
        filtered_elements.append(el)

    selected_elements = filtered_elements[:200]  # 只取前200个最优质的元素
    
    # 重新按 index 排序，方便 GPT 理解 DOM 顺序
    selected_elements.sort(key=lambda x: x['index'])

    gpt_text_lines = []
    valid_indices = []
    for i, el in enumerate(selected_elements):
        gpt_text_lines.append(f"{{id: {i}}} <{el['tagName']}> \"{el['text']}\"")
        valid_indices.append(el['index']) 
    
    return "\n".join(gpt_text_lines), valid_indices


def call_gpt_via_client(client, messages, model_name, allow_image_fallback=True):
    """
    统一调用接口
    :param allow_image_fallback: 如果因为图片导致 Refusal，是否允许移除图片后重试
    """
    def _extract_text_only(msgs):
        """辅助函数：从消息中移除图片，只保留文本"""
        text_msgs = []
        for m in msgs:
            # 过滤 content list，只留 type='text'
            if isinstance(m.get('content'), list):
                new_content = [item for item in m['content'] if item.get('type') == 'text']
                if new_content:
                    text_msgs.append({"role": m['role'], "content": new_content})
            else:
                # 如果本来就是纯文本字符串，直接保留
                text_msgs.append(m)
        return text_msgs

    # 第 1 次尝试：带图正常请求
    try:
        response = client.request(
            messages=messages,
            model=model_name,
            temperature=0.7,
            max_completion_tokens=1024,
            json_mode=True
        )
        
        # 检查是否被拒绝 (OpenAI Refusal)
        refusal = getattr(response.message, 'refusal', None)
        if refusal:
            print(f"🛑 OpenAI Refusal (Image likely): {refusal}")
            # 如果不允许回退，直接返回 None
            if not allow_image_fallback:
                return None
            # 否则，进入下方的回退逻辑...
        else:
            return response.message.content

    except Exception as e:
        print(f"⚠️ GPT 首次调用异常: {e}")
        if not allow_image_fallback:
            return None

    # 第 2 次尝试：降级为纯文本
    if allow_image_fallback:
        print("🔄 尝试移除图片，进行纯文本重试 (Text-Only Fallback)...")
        text_messages = _extract_text_only(messages)
        
        try:
            response = client.request(
                messages=text_messages, # 用纯文本
                model=model_name,
                temperature=0.7,
                max_completion_tokens=1024,
                json_mode=True
            )
            if getattr(response.message, 'refusal', None):
                print(f"❌ 纯文本重试依然被拒绝: {response.message.refusal}")
                return None
                
            print("✅ 纯文本重试成功！")
            return response.message.content
        except Exception as e:
            print(f"❌ 纯文本重试失败: {e}")
            return None
            
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
        ground_truth_trajectory = []
        current_depth = 0
        session_clicked_texts = set() 
        consecutive_scroll_count = 0
        consecutive_error_count = 0

        # 辅助函数：判断报错是否是成功的跳转
        def is_navigating_error(e):
            msg = str(e).lower()
            return "context" in msg or "navigating" in msg or "closed" in msg or "destroyed" in msg

        while current_depth < max_depth:
            print(f"📍 [Depth {current_depth}] URL: {page.url}")
            screenshot_before = get_page_screenshot_np(page)
            
            # 黑名单逻辑
            current_blacklist = session_clicked_texts.union(global_root_blacklist) if current_depth == 0 else session_clicked_texts
            elements_text, valid_indices = get_interactive_elements(page, current_blacklist)

            if not elements_text: elements_text = "No elements found. Try Scroll or Go Back."

            # 1. 选择动作
            print("🤔 思考下一步...")
            messages = prompt_select_deep_link(
                url=page.url, 
                elements_text=elements_text, 
                screenshot=screenshot_before,
                current_depth=current_depth,
                max_depth=max_depth,
                history_summary=history_breadcrumbs
            )
            content = call_gpt_via_client(gpt_client, messages, model_name, allow_image_fallback=True)
            if not content: break

            # 解析逻辑 (支持所有动作)
            action_type = "click"
            thought = ""
            chosen_id = -1
            input_value = ""
            scroll_dir = "down"
            key_comb = ""
            
            try:
                choice = json.loads(content)
                # 兼容处理
                if "action" not in choice: 
                    for k in ["click", "type", "hover", "scroll", "press", "goto", "go_back", "go_forward", "stop", "none"]:
                        if k in choice:
                            choice = choice[k]
                            choice["action"] = k
                            break
                
                action_type = choice.get("action", "click").lower()
                thought = choice.get("thought", "")
                
                # ID 智能处理
                raw_id = choice.get("element_id")
                if raw_id is not None:
                    if isinstance(raw_id, int): chosen_id = raw_id
                    elif isinstance(raw_id, str) and raw_id.isdigit(): chosen_id = int(raw_id)
                    elif action_type == "type": # 输入框自动补救
                        import re
                        match = re.search(r'\{id: (\d+)\} <INPUT>', elements_text)
                        chosen_id = int(match.group(1)) if match else -1
                
                input_value = choice.get("value")
                scroll_dir = choice.get("direction", "down")
                key_comb = choice.get("key_comb")

                print(f"💭 Thought: {thought}")

            except Exception as e:
                print(f"⚠️ JSON 解析失败: {e}")
                break

            # 动作执行
            
            # --- Group A: 终止类动作 (STOP / NONE) ---
            if action_type in ["stop", "none"]:
                consecutive_error_count = 0
                print(f"🛑 Agent 决定结束任务: {action_type.upper()} ({input_value})")
                action_record = {"step": current_depth, "type": action_type, "desc": f"Finished: {input_value}", "thought": thought}
                ground_truth_trajectory.append(action_record)
                break 

            # --- Group B: 导航类动作 (SCROLL) ---
            elif action_type == "scroll":
                consecutive_error_count = 0
                
                if consecutive_scroll_count >= 3:
                    print(f"⚠️ 连续滚动 ({consecutive_scroll_count}) 触发限制。拦截操作。")
                    warning_msg = "SYSTEM WARNING: Scroll limit reached! You have scrolled too many times. You MUST select a visible element to CLICK, TYPE, HOVER or PRESS now."
                    history_breadcrumbs.append(warning_msg)
                    continue 

                print(f"👇 执行滚动: {scroll_dir}")
                delta = 800 if scroll_dir == "down" else -800
                page.evaluate(f"window.scrollBy(0, {delta})")
                time.sleep(1)
                
                consecutive_scroll_count += 1
                history_breadcrumbs.append(f"Scroll {scroll_dir}")
                ground_truth_trajectory.append({"step": current_depth, "type": "scroll", "desc": f"Scroll {scroll_dir}", "thought": thought})
                continue

            # 非滚动操作，重置滚动计数
            consecutive_scroll_count = 0

            # --- Group C: 页面导航 (GO_BACK / FORWARD / GOTO) ---
            if action_type == "go_back":
                consecutive_error_count = 0
                print("🔙 执行后退")
                page.go_back()
                try: page.wait_for_load_state("domcontentloaded", timeout=5000)
                except: pass
                ground_truth_trajectory.append({"step": current_depth, "type": "go_back", "desc": "Go Back", "thought": thought})
                history_breadcrumbs.append("Back")
                continue

            elif action_type == "go_forward":
                consecutive_error_count = 0
                print("🔜 执行前进")
                page.go_forward()
                try: page.wait_for_load_state("domcontentloaded", timeout=5000)
                except: pass
                ground_truth_trajectory.append({"step": current_depth, "type": "go_forward", "desc": "Go Forward", "thought": thought})
                continue

            elif action_type == "goto":
                consecutive_error_count = 0
                print(f"🔗 直接跳转 (GOTO): {input_value}")
                try: page.goto(input_value, timeout=60000)
                except: print("⚠️ GOTO 失败")
                ground_truth_trajectory.append({"step": current_depth, "type": "goto", "desc": f"GOTO {input_value}", "thought": thought})
                history_breadcrumbs.append(f"GOTO {input_value}")
                current_depth += 1
                continue

            # --- Group D: 全局键盘 (PRESS) ---
            elif action_type == "press":
                consecutive_error_count = 0
                key = key_comb or "Enter"
                print(f"🎹 键盘按键: {key}")
                page.keyboard.press(key)
                time.sleep(1)
                ground_truth_trajectory.append({"step": current_depth, "type": "press", "desc": f"Press {key}", "thought": thought})
                try: page.wait_for_load_state("networkidle", timeout=2000)
                except: pass
                continue

            # --- Group E: 元素交互 (CLICK / TYPE / HOVER) ---
            elif isinstance(chosen_id, int) and 0 <= chosen_id < len(valid_indices):
                consecutive_error_count = 0 
                
                old_pages_count = len(browser_context.pages)
                original_index = valid_indices[chosen_id]
                
                # 提取文本
                raw_lines = elements_text.split('\n')
                target_text_line = next((line for line in raw_lines if f"{{id: {chosen_id}}}" in line), "Unknown")
                clean_text = target_text_line.split('"', 1)[-1].rstrip('"') if '"' in target_text_line else "Element"
                
                desc = f"{action_type.upper()} '{clean_text}'"
                if action_type == "type": desc += f" value='{input_value}'"
                
                print(f"👆 {desc}")
                action_record = {
                    "step_index": current_depth,
                    "action_type": action_type,
                    "element_text": clean_text,
                    "input_value": input_value,
                    "description": desc,
                    "thought": thought
                }
                ground_truth_trajectory.append(action_record)
                history_breadcrumbs.append(desc)

                try:
                    selector_str = 'a, button, [role="button"], h3, h4, .title, .job-name, .name, input, textarea, select'
                    elements = page.query_selector_all(selector_str)
                    
                    if 0 <= original_index < len(elements):
                        target_el = elements[original_index]
                        try: target_el.scroll_into_view_if_needed(timeout=1000)
                        except: pass

                        # === HOVER ===
                        if action_type == "hover":
                            try:
                                target_el.hover(force=True)
                                time.sleep(1) 
                                print("   🖱️ 悬停完成")
                            except: print("   ⚠️ 悬停失败")
                        
                        # === TYPE ===
                        elif action_type == "type":
                            print(f"   ⌨️ 输入: '{input_value}'")
                            try:
                                target_el.click(force=True)
                                modifier = "Meta" if sys.platform == "darwin" else "Control"
                                page.keyboard.press(f"{modifier}+A")
                                page.keyboard.press("Backspace")
                                page.keyboard.type(input_value or "", delay=50)
                                page.keyboard.press("Enter")
                                try: page.wait_for_load_state("networkidle", timeout=3000)
                                except: pass
                            except Exception as e:
                                if is_navigating_error(e): print("   🌊 输入触发跳转 (视为成功)")
                                else: print(f"   ⚠️ 输入失败: {e}")

                        # === CLICK ===
                        else: 
                            click_success = False

                            # 策略 A: Popup
                            try:
                                with page.expect_popup(timeout=2000) as popup_info:
                                    target_el.click(force=True, timeout=3000)
                                page = popup_info.value
                                print("   🔀 捕获新标签页")
                                click_success = True
                            except Exception as e:
                                if is_navigating_error(e): 
                                    print("   🌊 点击触发跳转 (视为成功)")
                                    click_success = True
                                else:
                                    # 策略 B: 普通点击
                                    try:
                                        target_el.click(force=True, timeout=3000)
                                        click_success = True
                                    except Exception as e_b:
                                        if is_navigating_error(e_b):
                                            print("   🌊 点击触发跳转 (视为成功)")
                                            click_success = True
                                        else:
                                            # 策略 C: 文本点击
                                            print(f"   🛡️ 句柄失效，尝试文本点击: {clean_text}")
                                            try:
                                                page.get_by_text(clean_text, exact=False).first.click(force=True, timeout=3000)
                                                click_success = True
                                            except Exception as e_t:
                                                if is_navigating_error(e_t):
                                                    print("   🌊 点击触发跳转 (视为成功)")
                                                    click_success = True
                            
                            # 策略 D: JS 强行点击 (带页面死亡检测)
                            if not click_success:
                                try:
                                    page.evaluate("1+1", timeout=500) # 检查页面存活
                                except:
                                    print("   🌊 (检查发现页面已失去响应，判定之前的点击成功)")
                                    click_success = True

                                if not click_success:
                                    print("   🔧 尝试 JS 强行点击")
                                    try:
                                        safe_js = f"""
                                        (() => {{
                                            const els = document.querySelectorAll('{selector_str}');
                                            if (els.length === 0) return "ZERO";
                                            const el = els[{original_index}];
                                            if (el) {{ el.click(); return "CLICKED"; }}
                                            return "NOT_FOUND";
                                        }})()
                                        """
                                        res = page.evaluate(safe_js)
                                        if res == "CLICKED" or res == "ZERO": click_success = True
                                    except Exception as e_js:
                                        if is_navigating_error(e_js) or "Total: 0" in str(e_js):
                                            print("   🌊 JS点击触发跳转 (视为成功)")
                                            click_success = True
                                        else:
                                            print(f"   ❌ 点击彻底失败: {e_js}")
                                            break

                    else:
                        print("⚠️ 元素 Index 失效")

                    # 后处理
                    try: page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except: pass
                    if len(browser_context.pages) > old_pages_count: page = browser_context.pages[-1]

                    # 登录页检测
                    try:
                        if "login" in page.title().lower() or "sign in" in page.title().lower():
                            print("🛑 遇到登录页，停止。")
                            break
                    except: pass

                    # 意图演化 (使用安全调用)
                    if action_type in ["click", "type", "hover", "press"]:
                        screenshot_after = get_page_screenshot_np(page)
                        evolve_msgs = prompt_evolve_task_description(
                            url=page.url, 
                            prev_task=current_task_desc, 
                            action_record=action_record,
                            screenshot_before=screenshot_before,
                            screenshot_after=screenshot_after
                        )
                        evolve_content = call_gpt_via_client(gpt_client, evolve_msgs, model_name, allow_image_fallback=True)
                        if evolve_content:
                            try:
                                new_task = json.loads(evolve_content).get("updated_task")
                                if new_task:
                                    print(f"✨ 新意图: {new_task}")
                                    current_task_desc = new_task
                                    img_path = save_screenshot_to_disk(screenshot_before, output_dir)
                                    evolution_history.append({
                                        "step": current_depth,
                                        "action": desc,
                                        "refined_task": new_task,
                                        "screenshot": img_path
                                    })
                            except: pass

                    current_depth += 1

                except Exception as e:
                    print(f"⚠️ 执行异常: {e}")
                    break

            # ID 无效的处理
            else:
                consecutive_error_count += 1
                print(f"⚠️ 无效的 ID (连续第 {consecutive_error_count} 次)，让 Agent 重试...")

                if consecutive_error_count >= 3:
                    print("🛑 连续多次 ID 无效，Agent 陷入死循环，强制结束任务。")
                    break 

                warning_msg = "SYSTEM ERROR: The element_id you selected is INVALID..."
                history_breadcrumbs.append(warning_msg)
                continue

        # 任务生成与保存
        print("🧠 生成最终任务...")
        final_screenshot = get_page_screenshot_np(page)
        if final_screenshot is None:
            final_screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        gen_messages = prompt_reverse_engineer_task(page.url, final_screenshot, history_breadcrumbs)
        content = call_gpt_via_client(gpt_client, gen_messages, model_name, allow_image_fallback=True)
        
        complex_task = None
        visual_evidence = []
        if content:
            try:
                res = json.loads(content)
                complex_task = res.get("complex_task")
                visual_evidence = res.get("visual_evidence", [])
            except: pass
        
        if not complex_task:
            complex_task = current_task_desc

        print(f"✅ 任务生成: {complex_task}\n")

        # 审计逻辑
        audit_status = False 
        audit_reason = "Audit Failed or Skipped"
        evidence_str = "\n".join(visual_evidence) if visual_evidence else "None"

        if complex_task and ground_truth_trajectory:
            print("🕵️ [Audit] 正在审计...")
            try:
                audit_msgs = prompt_audit_trajectory(complex_task, ground_truth_trajectory, evidence_str)
                audit_content = call_gpt_via_client(gpt_client, audit_msgs, model_name)
                if audit_content:
                    audit_res = json.loads(audit_content)
                    if audit_res.get("is_valid"):
                        audit_status = True
                        audit_reason = audit_res.get('reason')
                        print(f"✅ Audit Approved: {audit_reason}")
                    else:
                        audit_status = False
                        audit_reason = audit_res.get('reason')
                        print(f"⚠️ Audit Rejected: {audit_reason}")
            except Exception as e:
                print(f"Audit Error: {e}")
        
        final_data = {
            "task": complex_task,
            "start_url": start_url,
            "sites": ["custom"], 
            "final_url": page.url,
            "evolution_trace": evolution_history,
            "ground_truth_trajectory": ground_truth_trajectory,
            "visual_evidence": visual_evidence, 
            "audit_status": audit_status,
            "audit_reason": audit_reason,
        }
        
        print(f"💾 任务数据已构建，准备返回 (Status: {audit_status})")
        return final_data

    except Exception as e:
        print(f"❌ 任务彻底失败 (Exception): {e}")
        return None
    finally:
        page.close()


def main():
    parser = argparse.ArgumentParser(description="Evolutionary Task Generator (Step 1)")
    parser.add_argument("--target_start_url", type=str, required=True, help="Start URL")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of tasks to generate")
    parser.add_argument("--max_depth", type=int, default=5, help="Exploration depth")
    parser.add_argument("--output_dir", type=str, default="outputs/synthagent/custom_evolution", help="Output directory")
    
    # API 参数
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--openai_api_base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model", type=str, default="gpt-4o")
    
    args = parser.parse_args()

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%y-%m-%d-%H_%M_%S")
    final_output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 定义三个输出文件
    sys.stdout = Logger(os.path.join(final_output_dir, "run.log"))
    success_file = os.path.join(final_output_dir, "tasks_success.jsonl")
    rejected_file = os.path.join(final_output_dir, "tasks_rejected.jsonl")
    raw_trace_file = os.path.join(final_output_dir, "tasks_raw_trace.jsonl")

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
            headless=False, # 调试模式
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
                # 0. 更新全局去重黑名单
                if task_data.get("ground_truth_trajectory"):
                    try:
                        first_step = task_data["ground_truth_trajectory"][0]
                        first_text = first_step.get("element_text", "")
                        if first_text:
                            global_root_blacklist.add(first_text)
                            print(f"🚫 Added to Global Blacklist: {first_text}")
                    except: pass

                # 1. 先保存一份原始数据
                task_data["timestamp"] = datetime.now().isoformat()
                with open(raw_trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(task_data, ensure_ascii=False) + "\n")
                print(f"📦 [BACKUP] 原始数据已备份至 {raw_trace_file}")

                # 2. 根据规则进行质量分流
                is_high_quality = False
                trajectory = task_data.get("ground_truth_trajectory", [])
                task_desc = task_data.get("task")
                visual_ev = task_data.get("visual_evidence")

                # 是否精品的判定标准
                # 1. 任务描述必须存在 (task is not None/Empty)
                # 2. 轨迹长度 >= 2 (避免只有 1 步，长度过短)
                # 3. GPT 审计必须通过 (audit_status is True)
                if task_desc and len(trajectory) >= 2 and task_data.get("audit_status") is True:
                    is_high_quality = True
                else:
                    filter_reason = []
                    if not task_desc: filter_reason.append("No Task Desc")
                    if len(trajectory) < 2: filter_reason.append("Too Short")
                    if not visual_ev: filter_reason.append("No Evidence")
                    if not task_data.get("audit_status"): filter_reason.append(f"Audit Failed: {task_data.get('audit_reason')}")
                    
                    task_data["filter_reason"] = ", ".join(filter_reason)
                
                if is_high_quality: # ✅ 成功，精品
                    clean_data = {k:v for k,v in task_data.items() if k not in ["audit_status", "audit_reason"]}
                    with open(success_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(clean_data, ensure_ascii=False) + "\n")
                    print(f"🌟 [SUCCESS] 精品数据已保存至 {success_file} (Steps: {len(trajectory)})")
                else: # ❌ 失败，瑕疵
                    filter_reason = []
                    
                    # 1. 检查基础指标
                    if not task_desc: filter_reason.append("No Task Desc")
                    if len(trajectory) < 2: filter_reason.append("Too Short")
                    if not visual_ev: filter_reason.append("No Visual Evidence")
                    
                    # 2. 检查审计结果，把具体的拒绝原因加进去
                    if task_data.get("audit_status") is not True:
                        raw_reason = task_data.get("audit_reason", "Unknown Reason")
                        filter_reason.append(f"Audit Rejected Reason: {raw_reason}")
                    
                    task_data["filter_reason"] = " | ".join(filter_reason)
                    
                    with open(rejected_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(task_data, ensure_ascii=False) + "\n")
                    print(f"⚠️ [REJECTED] 瑕疵数据已保存至 {rejected_file}")
            
            else:
                print("❌ [FAILED] 任务执行过程中崩溃或被主动丢弃，无数据产生。")

        print(f"\n🎉 所有任务生成完毕！")
        print(f"   ✅ 精品数据: {success_file}")
        print(f"   📉 瑕疵数据: {rejected_file}")
        print(f"   📦 底稿备份: {raw_trace_file}")
        
        # 打印 GPT Token 使用统计
        print("\n📊 Token Usage:")
        print(gpt_client.token_usage)


if __name__ == "__main__":
    main()