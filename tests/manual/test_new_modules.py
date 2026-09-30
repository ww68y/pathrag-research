#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 V136-V143 新模块导入"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def test_module(name, module_path, func_name):
    """测试单个模块导入"""
    try:
        module = __import__(module_path, fromlist=[func_name])
        func = getattr(module, func_name)
        print(f"✅ {name}: 导入成功 - {func_name}")
        return True
    except Exception as e:
        print(f"❌ {name}: 失败 - {str(e)[:100]}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("测试 V136-V143 新模块导入")
    print("=" * 60)
    
    modules = [
        ("V136 多模态融合", "api.features.chats.multimodal_knowledge_fusion", "build_multimodal_evidence_fusion"),
        ("V137 时序推理", "api.features.chats.temporal_reasoning_enhancement", "build_temporal_reasoning_enhancement"),
        ("V138 多语言对齐", "api.features.chats.cross_lingual_alignment", "build_cross_lingual_alignment"),
        ("V139 隐私保护", "api.features.chats.privacy_preserving_retrieval", "build_privacy_preserving_retrieval"),
        ("V140 可解释性", "api.features.chats.explainability_dashboard", "build_explainability_dashboard_data"),
        ("V141 RL优化器v2", "api.features.chats.reinforcement_learning_optimizer_v2", "build_rl_optimizer_v2"),
        ("V142 增量更新", "api.features.chats.incremental_graph_update", "build_incremental_update_pipeline"),
        ("V143 边缘部署", "api.features.chats.edge_deployment_support", "build_edge_deployment_support"),
    ]
    
    results = []
    for name, path, func in modules:
        ok = test_module(name, path, func)
        results.append((name, ok))
    
    print()
    print("=" * 60)
    success = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"结果: {success}/{total} 个模块导入成功")
    
    if success == total:
        print("🎉 所有新模块验证通过！")
        sys.exit(0)
    else:
        failed = [name for name, ok in results if not ok]
        print(f"⚠️ 失败的模块: {', '.join(failed)}")
        sys.exit(1)
