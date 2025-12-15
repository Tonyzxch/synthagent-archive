from syn.tools import tools_ndarray_to_base64_image

# 1. 选择目标 Prompt
def prompt_select_deep_link(
    url: str, 
    elements_text: str, 
    screenshot: "np.ndarray" = None,
    current_depth: int = 0,
    max_depth: int = 3
) -> list[dict]:
    # 1. 根据当前深度动态生成策略
    # 将任务进程归一化为 0.0 到 1.0 的进度条
    progress_ratio = current_depth / max(max_depth, 1)

    if progress_ratio < 0.35:
        # 【早期阶段】：寻找入口
        phase_instruction = """
        **Current Phase: BROAD NAVIGATION (Exploration)**
        - You are likely on a Homepage, Index, or Portal.
        - **Goal**: Enter a specific domain or category. Move from "Root" to "Branch".
        - **Target**: High-level categories (e.g., "Departments", "Topics", "Services", "Directories").
        - **Avoid**: Specific individual items (unless they are the ONLY option), generic footers (About, Contact).
        - **Heuristic**: Choose links that act as "Containers" for other content.
        """
    elif progress_ratio < 0.7:
        # 【中期阶段】：过滤与细化
        phase_instruction = """
        **Current Phase: REFINEMENT & FILTERING (Narrowing Down)**
        - You are likely on a List Page or Sub-category.
        - **Goal**: Narrow the search space to find high-value targets.
        - **Target**: 
            1. Sub-categories (e.g., "Smartphones" inside "Electronics").
            2. Filters/Attributes (e.g., "Sort by Date", "Price Range", "Status").
            3. Specific Lists (e.g., "Top Rated", "Newest Arrivals").
        - **Avoid**: Going back to the Homepage.
        """
    else:
        # 【后期阶段】：锁定实体
        phase_instruction = """
        **Current Phase: TARGET ACQUISITION (Selection)**
        - You are deep in the navigation structure.
        - **Goal**: Select a specific **Leaf Node** (The actual content).
        - **Target**: The specific Entity/Item/Article/File. 
            - Look for the element with the **richest detail** (longest title, specific price, dates, unique ID).
            - It is often the largest clickable card or text block in the main viewport.
        - **Avoid**: Top-level navigation bars (Home, Menu) that reset your progress.
        """

    # 2. 构建通用 Prompt
    prompt = f"""You are an Autonomous Information Seeker exploring a website structure.
Current Context:
- URL: {url}
- Progress: Step {current_depth + 1} of {max_depth} (Phase: {int(progress_ratio * 100)}% Complete)

{phase_instruction}

**Universal Selection Logic (Abstract Rules)**

1. **Analyze the DOM Tree**:
   - **Root Nodes (Main Menu)**: Usually at the top. Click these ONLY if you are at Step 1.
   - **Branch Nodes (Categories/Lists)**: Group similar items. Good for middle steps.
   - **Leaf Nodes (Entities)**: The final destination (Product, Article, Person, Job, File). Good for final steps.

2. **Visual & Textual Heuristics**:
   - **Density**: If a link is surrounded by data (dates, numbers, status), it is likely a specific Entity (Leaf).
   - **Uniqueness**: "Category A" is a Branch. "Item #9527 with specs X" is a Leaf.
   - **Promptness**: "See All", "More", "Next Page" are navigational actions (valid for middle steps).

3. **Scroll Logic**:
   - If the main content area is empty or obscured by cookie banners/popups, or if you only see generic headers -> **Return -1**.

**Available Interactive Elements (Index | Tag | Text)**:
{elements_text}

**Output Requirement**
Return JSON ONLY: {{"element_id": <int_id>}} 
(Return -1 to scroll down)
"""
    
    content = [{"type": "text", "text": prompt}]
    
    if screenshot is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": tools_ndarray_to_base64_image(screenshot)},
        })

    return [{"role": "user", "content": content}]


# 2. 任务进化 Prompt
def prompt_evolve_task_description(
    url: str,
    prev_task: str,
    action_desc: str,
    screenshot_before: "np.ndarray",
    screenshot_after: "np.ndarray"
) -> list[dict]:
    prompt = f"""You are a "User Intent Simulator".
You have access to the screen state BEFORE and AFTER an action.
Your goal is to infer the specific intent that led to this action.

**Context**
- Previous Intent: "{prev_task}"
- Action Taken: {action_desc}

**Input Data**
- Image 1: The screen BEFORE the action.
- Image 2: The screen AFTER the action (Result).

**Instructions**
1. **Analyze the Change**: Look at Image 2 to understand what the action actually accomplished.
2. **Formulate Intent**: Rewrite the "Previous Intent" to include this new step.
3. **CRITICAL RULE: NO INFORMATION LEAKAGE**:
   - The intent must be actionable based ONLY on information visible in **Image 1**.
   - Do NOT describe hidden attributes that are only revealed in Image 2.
   - *Example (Bad)*: "Click the link to see the $50 price" (If price is only on pg 2).
   - *Example (Good)*: "Click the link to check the price."

**Output Requirement**
Return JSON ONLY: {{ "updated_task": "The natural language user intent string..." }}
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
    if not history_summary:
        history_str = "Homepage"
    else:
        history_str = " -> ".join(history_summary)
    
    prompt = f"""You are an Expert Task Generator for a Web Agent Benchmark.
The user has navigated through: {history_str}
Current URL: {url}

**Goal**
Generate a **High-Level User Intent** based on the navigation history and the final page context.
The "Less is More" Rule: Real users are lazy. They usually have **1 major constraint** or **1-2 specific feature requirements**, NEVER list too many specs at once.
CRITICAL: The task must be **challenging but solvable**. Avoid "Over-Specification" (describing the specific entity in the screenshot too precisely).

**Universal Constraint Logic**
1. **Hard Constraints (From History)**: 
   - If the user clicked specific filters, tabs, or sort buttons (e.g., "Sort by Date", "Label: Bug", "Price: Low to High"), these conditions should be included in the task.
   
2. **Soft Constraints (From Screenshot - Generalization)**: 
   - If you see a specific entity (e.g., a specific product, a specific post, a specific code issue). 
   - Describe it by its **distinguishing features** (Less is More).
   - **Do NOT** describe everything about its features. Avoid to be too specific.

**Output Requirement**
Return JSON ONLY: 
{{
    "complex_task": "The generated instruction string..."
}}
"""
    
    message = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", 
             "image_url": {"url": tools_ndarray_to_base64_image(screenshot)}}
        ]}
    ]
    return message