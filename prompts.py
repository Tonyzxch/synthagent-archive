from syn.tools import tools_ndarray_to_base64_image

# 1. 动作选择 Prompt
def prompt_select_deep_link(
    url: str, 
    elements_text: str, 
    screenshot: "np.ndarray" = None,
    current_depth: int = 0,
    max_depth: int = 5,
    history_summary: list[str] = []
) -> list[dict]:
    
    progress_ratio = current_depth / max(max_depth, 1)
    
    if progress_ratio < 0.3:
        phase_instruction = "**Phase: DISCOVERY**\nGoal: Broad search or Category selection. Use `TYPE` (Search) or `CLICK` menus."
    elif progress_ratio < 0.7:
        phase_instruction = "**Phase: NAVIGATION**\nGoal: Locate content. `SCROLL` to find items, `CLICK` filters, or `GO_BACK` if stuck."
    else:
        phase_instruction = "**Phase: TARGETING**\nGoal: Interact with a specific Entity. `CLICK` the target. If task is done, `NONE`."

    history_str = " -> ".join(history_summary[-5:]) # 目前只保留最近5步操作，如果深度加深需注意这里

    prompt = f"""You are an Autonomous Web Agent exploring a website.
**Context**:
- URL: {url}
- History: {history_str}
{phase_instruction}

**Full Standard Action Space**:
1. **CLICK**: Select a link/button. (Param: `element_id`)
2. **TYPE**: Input text. (Param: `element_id`, `value`)
3. **HOVER**: Mouse over an element. (Param: `element_id`)
4. **SCROLL**: Move page. (Param: `direction`: 'up' or 'down')
5. **PRESS**: Keyboard shortcut. (Param: `key_comb`: 'Enter', 'PageDown', 'Tab')
6. **GO_BACK**: Return to previous page. (Param: None)
7. **GO_FORWARD**: Go to next page (rare). (Param: None)
8. **GOTO**: Direct URL navigation. (Param: `value` as url). *Use only if navigation is broken.*
9. **STOP**: Give up. (Param: `value` as reason). *Use only if task is impossible.*
10. **NONE**: Finish task. (Param: `value` as summary). *Use when goal is reached.*

**Interactive Elements**:
{elements_text}

**Output Format (JSON Only)**:
{{
    "thought": "Reasoning step...",
    "action": "click" | "type" | "hover" | "scroll" | "press" | "goto" | "go_back" | "go_forward" | "stop" | "none",
    "element_id": <int> or null,
    "value": "<string>" or null,
    "direction": "down" or null,
    "key_comb": "<string>" or null
}}
"""
    content = [{"type": "text", "text": prompt}]
    if screenshot is not None:
        content.append({"type": "image_url", "image_url": {"url": tools_ndarray_to_base64_image(screenshot)}})
    return [{"role": "user", "content": content}]


# 2. 任务演化 Prompt
def prompt_evolve_task_description(
    url: str,
    prev_task: str,
    action_record: dict,
    screenshot_before: "np.ndarray",
    screenshot_after: "np.ndarray"
) -> list[dict]:
    
    # 格式化动作描述
    act_type = action_record.get('action_type', 'unknown')
    if act_type == 'scroll':
        action_desc = f"Scrolled {action_record.get('element_text', 'down')} to explore"
    elif act_type == 'go_back':
        action_desc = "Went back to previous page"
    elif act_type == 'type':
        action_desc = f"Typed '{action_record.get('input_value')}' into search"
    else:
        action_desc = f"Clicked link '{action_record.get('element_text')}'"

    thought = action_record.get('thought', 'No thought recorded')

    prompt = f"""You are a "User Intent Simulator".
Analyze the user's latest action to refine their High-Level Goal.

**Context**
- **Previous Intent**: "{prev_task}"
- **User's Thought**: "{thought}"
- **Action Taken**: {action_desc}
- **Visual Change**: (Compare Image 1 and Image 2)

**Refinement Logic**:
1. **Search**: If they typed a query, the intent is now specific to that topic.
2. **Navigation**: If they clicked a specific category, narrow the intent.
3. **Exploration (Scroll)**: If they scrolled, they are "browsing" or "looking for more info". Keep the intent broad but imply thoroughness.
4. **Correction (Back)**: If they went back, ignore the previous dead-end.

**Output Requirement**:
Return JSON ONLY: {{ "updated_task": "Natural language intent string..." }}
"""
    
    content = [{"type": "text", "text": prompt}]
    if screenshot_before is not None:
        content.append({"type": "image_url", "image_url": {"url": tools_ndarray_to_base64_image(screenshot_before)}})
    if screenshot_after is not None:
        content.append({"type": "image_url", "image_url": {"url": tools_ndarray_to_base64_image(screenshot_after)}})

    return [{"role": "user", "content": content}]


# 3. 逆向工程 Prompt
def prompt_reverse_engineer_task(
    url: str, 
    screenshot: "np.ndarray", 
    history_summary: list[str]
) -> list[dict]:
    
    history_str = " -> ".join(history_summary)
    
    prompt = f"""You are a Forensic Task Reconstructor.
Generate a natural, high-level user instruction based on the behavior history.

**User History**: {history_str}
**Final Page Context**: (See Image)

**Synthesis Rules**:

1.  **INPUT HONESTY (Strict)**: 
    - Check the history. Did the user use the `TYPE` action?
    - If YES (e.g., Typed "games"), your task MUST start with "Search for 'games'...", do NOT hallucinate specific terms they didn't type.

2.  **NOISE FILTERING (Important)**:
    - Ignore mechanical steps like `Scroll down`, `Hover`, or `Go Back` in the final task description unless they are the *only* actions taken.
    - Focus on the **Key Milestones** (Search -> Click Category -> Click Item).

3.  **TARGET GENERALIZATION**:
    - **Do NOT** copy the exact full text of the final element (e.g., "Five Nights at Freddy's: Secret of the Mimic (2025)").
    - **Do** describe it naturally (e.g., "find information about the 'Secret of the Mimic' game").

**Output JSON**:
{{
    "visual_evidence": ["Short text string from screen confirming success"],
    "complex_task": "The natural language instruction."
}}
"""
    content = [{"type": "text", "text": prompt}]
    if screenshot is not None:
        content.append({"type": "image_url", "image_url": {"url": tools_ndarray_to_base64_image(screenshot)}})
    return [{"role": "user", "content": content}]


# 4. 轨迹审计 Prompt
def prompt_audit_trajectory(
    task_description: str,
    trajectory: list[dict],
    visual_evidence: str = "None provided"
) -> list[dict]:
    
    # 提取最后一步动作
    last_step = trajectory[-1] if trajectory else {}
    last_action = last_step.get('description', 'Unknown')
    
    # 提取倒数第二步（辅助判断）
    prev_step = trajectory[-2] if len(trajectory) > 1 else {}
    prev_action = prev_step.get('description', 'None')

    prompt = f"""You are a Result-Oriented Task Auditor.
Your job is to determine if the User's **Final State** and **Last Action** successfully completed the Task.

**The Task**: "{task_description}"

**The Final Evidence**:
- **Last Action Performed**: "{last_action}"
- **Previous Action**: "{prev_action}"
- **Content Visible on Final Screen**: "{visual_evidence}"

**Audit Rules (Based on Task Type)**:

1. **Type A: Information Seeking (e.g., "Find the price", "Check the date")**
   - **Criteria**: Does the "Content Visible on Final Screen" contain the answer or the specific entity requested?
   - **Ignore**: Do NOT judge the search keywords used in early steps. As long as they reached the right page, it is a PASS.

2. **Type B: Operation/Navigation (e.g., "Navigate to...", "Click on...")**
   - **Criteria**: Does the "Last Action" directly interact with the target?
   - **Example**: If Task is "Find the Login Page", and Last Action is "Click 'Sign In'", it is a PASS.

**Judgment Logic**:
- If the final state matches the goal -> **true**.
- If the user is still searching or on an irrelevant page -> **false**.

**Output**: JSON
{{
    "is_valid": ..., 
    "reason": "Explain why the final outcome matches the task or not."
}}
"""
    return [{"role": "user", "content": prompt}]