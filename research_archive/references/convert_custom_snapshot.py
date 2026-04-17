import json
import os

# 配置你的 Step 1 输出路径 (请根据实际情况确认时间戳文件夹)
# 注意：优先使用 db.simplified.jsonl，因为它更小且包含最终状态
INPUT_FILE = "/home/zxch/SynthAgent_new/outputs/synthagent/custom/25-12-11-09_00_25/db.simplified.jsonl"
OUTPUT_FILE = "configs/synthagent_custom.jsonl"
TARGET_URL = "http://18.216.88.140:7770"  # 你的目标网站

def convert():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return

    tasks = []
    seen_tasks = set()
    
    print(f"📖 正在读取: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                # 只提取状态为 END (成功生成) 的任务
                if record.get('status') == 'END':
                    # 获取 High Level Task
                    hl_tasks = record.get('high_level_tasks', [])
                    if hl_tasks:
                        task_text = hl_tasks[-1]
                        
                        # 简单去重
                        if task_text not in seen_tasks:
                            tasks.append({
                                "task": task_text,
                                "start_url": TARGET_URL,
                                "sites": ["custom"]
                            })
                            seen_tasks.add(task_text)
            except Exception as e:
                print(f"⚠️ 跳过错误行: {e}")

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
            
    print(f"✅ 转换完成！共提取 {len(tasks)} 条唯一任务。")
    print(f"📂 保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    convert()