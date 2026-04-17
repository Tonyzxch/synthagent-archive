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
        phase_instruction = "**Phase: EXPLORATION**\nGoal: Start exploration. Use `TYPE` for specific searches or `CLICK` on broad categories/menus."
    elif progress_ratio < 0.7:
        phase_instruction = "**Phase: NAVIGATION & FILTERING**\nGoal: Locate specific content. \n- Prioritize `CLICK` on Filters, Sorting options, or Sub-categories to narrow down results.\n- Use `SCROLL` to find items if not visible."
    else:
        phase_instruction = "**Phase: TARGETING & VERIFICATION**\nGoal: Interact with the final target.\n- **VERIFY**: Do NOT stop immediately after a click. Wait/Scroll to confirm the action succeeded (e.g., success message, page update) before using `NONE`."

    history_str = " -> ".join(history_summary[-5:]) 

    prompt = f"""You are an Autonomous Web Agent exploring a website.
**Context**:
- URL: {url}
- History: {history_str}
{phase_instruction}

**Full Standard Action Space**:
1. **CLICK**: Select a link/button. 
   - Param: `element_id`
   - Note: Do NOT use CLICK for input fields, use TYPE instead.
2. **TYPE**: Input text into a search bar or form.
   - **CRITICAL RULE**: If the element tag is `<INPUT>` or `<TEXTAREA>`, you MUST use `TYPE`. Do NOT use `CLICK`.
   - Param: `element_id`, `value` (the text to type)
3. **HOVER**: Mouse over an element. 
   - Param: `element_id`
   - Use ONLY for dropdown menus or revealing hidden options.
4. **SCROLL**: Move page. (Param: `direction`: 'up' or 'down')
5. **PRESS**: Keyboard shortcut. (Param: `key_comb`: 'Enter', 'PageDown', 'Tab')
6. **GO_BACK**: Return to previous page. (Param: None)
7. **GO_FORWARD**: Go to next page (rare). (Param: None)
8. **GOTO**: Direct URL navigation. (Param: `value` as url). *Use only if navigation is broken.*
9. **STOP**: Give up. (Param: `value` as reason). *Use only if task is impossible.*
10. **NONE**: Finish task. (Param: `value` as summary). 
    - **RESTRICTION**: Do NOT use `NONE` on a Search Results list, Category page, or Homepage.
    - **REQUIREMENT**: You MUST be on a specific **Item Detail Page** or have completed a specific action (e.g., Submission, Download, Purchase).

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
3. **Exploration**: If they scrolled/hovered, they are looking for details.

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
    
    prompt = f"""You are a Universal Web Task Architect.
Your goal is to reverse-engineer a **Specific, Constraint-Based User Instruction** by analyzing the functional logic of the user's behavior.

**Context**:
- **User History**: {history_str}
- **Final Page Context**: (See Image)

**Core Philosophy**:
Trust your reasoning. Do not describe *what* the user clicked, but **WHY** they clicked it. Convert the "Mechanism" (Action) into a "Constraint" (Task Intent).

**Generation Rules (Strictly Follow)**:

1.  **TASK GENERATION - "INTENT INFERENCE"**: 
    - **Logic**: Translate the user's interactions into **specific semantic constraints** within the Task description.
    - **Refinement Actions**: If the user selected a subset of items (via filters/menus), the Task MUST specify the **scope** (e.g., a specific timeframe, budget range, or attribute like color or size).
    - **Prioritization Actions**: If the user reordered items (via sorting), the Task MUST specify an **Optimization Goal** (e.g., "find the cheapest", "find the latest", "find the most affordable", "find the most relevant", etc.).
    - **Anti-Vagueness**: Never use phrases like "broad range" or "various options". If a user makes a selection, the task MUST be **specific**, not broad.

2.  **TIPS GENERATION - "MECHANISM MAPPING"**:
    - **Logic**: The Tip must explain the **Functional Utility** of the tool used, ensuring the user knows *how* to satisfy the constraint mentioned above.
    - **Structure**: "TIPS: Use the [UI Component Name] to [Strategic Benefit]."
    - **Constraint**: Do not mention specific data values (like numbers or proper nouns) in the Tip. Keep it focused on the **tool's capability**.

3.  **NO VERBATIM TITLES**: 
    - Describe the target by its **function, category, or content**. Do not copy the exact commercial title string.

4.  **INPUT HONESTY**: 
    - If user typed text, the Task must explicitly state "Search for [text]...", do not hallucinate terms.

**Output JSON ONLY**:
{{
    "visual_evidence": ["Short text string from screen verifying success"],
    "complex_task": "The instruction containing explicit semantic constraints derived from actions.",
    "tips": "The hint string explaining the tool's utility."
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
    
    last_step = trajectory[-1] if trajectory else {}
    last_action = last_step.get('description', 'Unknown')
    
    prompt = f"""You are a Lenient Task Evaluator.
Determine if the User's final state is **semantically aligned** with the Task.

**The Task**: "{task_description}"
**Last Action**: "{last_action}"
**Final Visual Evidence**: "{visual_evidence}"

**Evaluation Logic (Lenient)**:
1. **Semantic Relevance**: Is the user on a page *relevant* to the task? (e.g., If the task is "Find X", and the user is on the detail page of X, it is a PASS).
2. **Action Consistency**: Did the user's actions lead them here reasonably?
3. **Implicit Success**: If the user found the correct item/page but didn't explicitly perform the final click (e.g., "Submit", "Download", "Buy", "Play"), but the target is clearly visible and correct, count it as **VALID** for exploration/navigation tasks.

**Output JSON**:
{{
    "is_valid": true/false, 
    "reason": "Brief explanation."
}}
"""
    return [{"role": "user", "content": prompt}]