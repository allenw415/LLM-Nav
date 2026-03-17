from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from controller import PerceptionExecutionController
from detector.detector import PerceptionDetector
from executor.executor import ActionExecutor
from renderer.renderer import StreetViewRenderer
from renderer.viewer import ImageViewer


# =========================
# 基本設定
# =========================
API_KEY = "YOUR_GOOGLE_API_KEY"
PANOS_JSON_PATH = "panos.json"
RUNTIME_VIEW_DIR = "runtime_views"
VIEWER_PORT = 5000


# =========================
# 載入 graph
# =========================
def load_graph(json_path: str | Path) -> Dict[str, Any]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Cannot find graph file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    if not isinstance(graph, dict):
        raise ValueError("panos.json must be a dict mapping panoID -> node_info")

    return graph


# =========================
# graph 檢查
# =========================
def summarize_graph(graph: Dict[str, Any]) -> None:
    num_nodes = len(graph)
    num_links = 0
    num_nodes_with_links = 0
    num_broken_links = 0

    floors = {}

    for pano_id, node in graph.items():
        links = node.get("links", [])
        if links:
            num_nodes_with_links += 1
        num_links += len(links)

        floor = str(node.get("floor", "unknown"))
        floors[floor] = floors.get(floor, 0) + 1

        for link in links:
            nxt = link.get("panoID")
            if nxt not in graph:
                num_broken_links += 1

    print("=" * 60)
    print("Graph Summary")
    print("=" * 60)
    print(f"Total nodes             : {num_nodes}")
    print(f"Total links             : {num_links}")
    print(f"Nodes with links        : {num_nodes_with_links}")
    print(f"Broken links            : {num_broken_links}")
    print(f"Floors distribution     : {floors}")
    print("=" * 60)


def get_valid_start_nodes(graph: Dict[str, Any]) -> List[str]:
    """
    找出至少有一條 link 的起點，較適合做移動測試
    """
    valid = []
    for pano_id, node in graph.items():
        links = node.get("links", [])
        if isinstance(links, list) and len(links) > 0:
            valid.append(pano_id)
    return valid


def validate_start_pano(graph: Dict[str, Any], pano_id: str) -> None:
    if pano_id not in graph:
        raise ValueError(f"Start panoID not found in graph: {pano_id}")


# =========================
# state 建立
# =========================
def make_initial_state(graph: Dict[str, Any], pano_id: str) -> Dict[str, Any]:
    node = graph[pano_id]
    return {
        "panoID": pano_id,
        "heading": 0.0,
        "pitch": 0.0,
        "zoom": 1,
        "floor": node.get("floor", "unknown"),
        "last_action": None,
    }


# =========================
# 初始化系統
# =========================
def build_system(graph: Dict[str, Any]) -> PerceptionExecutionController:
    renderer = StreetViewRenderer(api_key=API_KEY)
    detector = PerceptionDetector()
    executor = ActionExecutor(graph=graph)
    viewer = ImageViewer(image_dir=RUNTIME_VIEW_DIR, port=VIEWER_PORT)

    controller = PerceptionExecutionController(
        renderer=renderer,
        detector=detector,
        executor=executor,
        viewer=viewer,
    )
    return controller


# =========================
# 單步測試
# =========================
def test_single_step(controller: PerceptionExecutionController, state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n[TEST] Single Step")
    print(f"Start state: {state}")

    new_state, observation, action = controller.step(state)

    print(f"Action      : {action}")
    print(f"New state   : {new_state}")
    print(f"Confidence  : {observation.get('confidence')}")
    print(f"Scene desc  : {observation.get('scene_desc')}")
    print(f"Landmarks   : {len(observation.get('landmarks', []))}")
    print(f"OCR texts   : {len(observation.get('ocr_texts', []))}")

    return new_state


# =========================
# 多步連續測試
# =========================
def test_multi_steps(
    controller: PerceptionExecutionController,
    state: Dict[str, Any],
    steps: int = 5,
    sleep_sec: float = 1.0,
) -> Dict[str, Any]:
    print(f"\n[TEST] Multi Steps: {steps}")

    current_state = state
    for i in range(steps):
        print(f"\n--- step {i} ---")
        try:
            current_state, observation, action = controller.step(current_state)
            print(f"Action      : {action}")
            print(f"Current pano: {current_state['panoID']}")
            print(f"Heading     : {current_state['heading']}")
            print(f"Pitch       : {current_state['pitch']}")
            print(f"Zoom        : {current_state['zoom']}")
            print(f"Confidence  : {observation.get('confidence')}")
            print(f"Scene desc  : {observation.get('scene_desc')}")
        except Exception as e:
            print(f"[ERROR at step {i}] {e}")
            break

        time.sleep(sleep_sec)

    return current_state


# =========================
# 指定起點測試
# =========================
def test_specific_start(
    controller: PerceptionExecutionController,
    graph: Dict[str, Any],
    start_pano_id: str,
    steps: int = 3,
) -> None:
    print("\n[TEST] Specific Start")
    validate_start_pano(graph, start_pano_id)
    state = make_initial_state(graph, start_pano_id)
    test_multi_steps(controller, state, steps=steps)


# =========================
# 隨機起點測試
# =========================
def test_random_start(
    controller: PerceptionExecutionController,
    graph: Dict[str, Any],
    steps: int = 3,
) -> None:
    print("\n[TEST] Random Start")
    valid_nodes = get_valid_start_nodes(graph)
    if not valid_nodes:
        raise RuntimeError("No valid start nodes with links found.")

    start_pano_id = random.choice(valid_nodes)
    print(f"Random start panoID: {start_pano_id}")
    state = make_initial_state(graph, start_pano_id)
    test_multi_steps(controller, state, steps=steps)


# =========================
# 多起點批次測試
# =========================
def test_batch_starts(
    controller: PerceptionExecutionController,
    graph: Dict[str, Any],
    num_starts: int = 5,
    steps_per_start: int = 2,
) -> None:
    print("\n[TEST] Batch Starts")

    valid_nodes = get_valid_start_nodes(graph)
    if not valid_nodes:
        raise RuntimeError("No valid start nodes with links found.")

    sample_nodes = random.sample(valid_nodes, min(num_starts, len(valid_nodes)))

    for idx, pano_id in enumerate(sample_nodes):
        print(f"\n========== Batch Case {idx + 1} / {len(sample_nodes)} ==========")
        print(f"Start panoID: {pano_id}")
        state = make_initial_state(graph, pano_id)

        try:
            test_multi_steps(controller, state, steps=steps_per_start, sleep_sec=0.5)
        except Exception as e:
            print(f"[BATCH ERROR] panoID={pano_id}, error={e}")


# =========================
# executor 純測試
# 不依賴 API，先檢查 graph 移動邏輯
# =========================
def test_executor_only(graph: Dict[str, Any], num_cases: int = 5) -> None:
    print("\n[TEST] Executor Only")

    executor = ActionExecutor(graph=graph)
    valid_nodes = get_valid_start_nodes(graph)
    if not valid_nodes:
        raise RuntimeError("No valid start nodes with links found.")

    sample_nodes = random.sample(valid_nodes, min(num_cases, len(valid_nodes)))

    for pano_id in sample_nodes:
        state = make_initial_state(graph, pano_id)
        node = graph[pano_id]
        links = node.get("links", [])
        print(f"\nStart panoID: {pano_id}")
        print(f"Links count : {len(links)}")

        # 測試 turn
        s1 = executor.execute(state, {"type": "TURN_RIGHT", "value": 30})
        print(f"After TURN_RIGHT 30 -> heading={s1['heading']}")

        # 測試 zoom
        s2 = executor.execute(s1, {"type": "ZOOM_IN", "value": 1})
        print(f"After ZOOM_IN 1    -> zoom={s2['zoom']}")

        # 測試 move
        s3 = executor.execute(s2, {"type": "MOVE_TO_LINK", "value": 0})
        print(f"After MOVE_TO_LINK -> panoID={s3['panoID']}, floor={s3['floor']}")


# =========================
# renderer 純測試
# =========================
def test_renderer_only(graph: Dict[str, Any], pano_id: str) -> None:
    print("\n[TEST] Renderer Only")

    renderer = StreetViewRenderer(api_key=API_KEY)
    state = make_initial_state(graph, pano_id)

    rendered = renderer.render_view(state)
    print("Rendered result:")
    print(rendered)

    rendered_four = renderer.render_multi_view(state)
    print(f"Rendered 4 views count: {len(rendered_four['views'])}")


# =========================
# 自訂 policy 測試
# =========================
def simple_policy(state: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    一個簡單測試版 policy：
    1. 若目前 zoom 還小，先 zoom in
    2. 若沒有 landmarks / OCR，轉右
    3. 否則往第一條 link 前進
    """
    if int(state.get("zoom", 1)) < 2:
        return {"type": "ZOOM_IN", "value": 1}

    if not observation.get("landmarks") and not observation.get("ocr_texts"):
        return {"type": "TURN_RIGHT", "value": 45}

    return {"type": "MOVE_TO_LINK", "value": 0}


def test_custom_policy(
    controller: PerceptionExecutionController,
    graph: Dict[str, Any],
    steps: int = 5,
) -> None:
    print("\n[TEST] Custom Policy")
    valid_nodes = get_valid_start_nodes(graph)
    if not valid_nodes:
        raise RuntimeError("No valid start nodes found.")

    start_pano_id = random.choice(valid_nodes)
    print(f"Start panoID: {start_pano_id}")

    state = make_initial_state(graph, start_pano_id)

    for i in range(steps):
        print(f"\n--- custom policy step {i} ---")
        try:
            state, observation, action = controller.step(state, policy_fn=simple_policy)
            print(f"Action      : {action}")
            print(f"Current pano: {state['panoID']}")
            print(f"Heading     : {state['heading']}")
            print(f"Zoom        : {state['zoom']}")
            print(f"Confidence  : {observation.get('confidence')}")
        except Exception as e:
            print(f"[CUSTOM POLICY ERROR] {e}")
            break
        time.sleep(0.8)


# =========================
# main
# =========================
def main() -> None:
    graph = load_graph(PANOS_JSON_PATH)
    summarize_graph(graph)

    valid_nodes = get_valid_start_nodes(graph)
    if not valid_nodes:
        raise RuntimeError("No valid start nodes with links found in panos.json")

    print(f"Valid start nodes: {len(valid_nodes)}")
    print(f"Viewer will run at: http://127.0.0.1:{VIEWER_PORT}")

    # 1. 純 graph / state / transition 測試
    test_executor_only(graph, num_cases=5)

    # 2. renderer 測試
    sample_start = random.choice(valid_nodes)
    test_renderer_only(graph, sample_start)

    # 3. 建立完整系統
    controller = build_system(graph)

    # 4. 單步測試
    init_state = make_initial_state(graph, sample_start)
    state = test_single_step(controller, init_state)

    # 5. 多步測試
    test_multi_steps(controller, state, steps=5, sleep_sec=1.0)

    # 6. 隨機起點測試
    test_random_start(controller, graph, steps=3)

    # 7. 指定起點測試
    test_specific_start(controller, graph, start_pano_id=sample_start, steps=3)

    # 8. 批次起點測試
    test_batch_starts(controller, graph, num_starts=5, steps_per_start=2)

    # 9. 自訂 policy 測試
    test_custom_policy(controller, graph, steps=5)

    print("\nAll tests finished.")


if __name__ == "__main__":
    main()