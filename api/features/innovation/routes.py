"""
V136-V143 创新模块 API 路由
===========================

提供 8 大创新领域的 RESTful API 接口：
- 多模态知识融合 (V136)
- 时序推理增强 (V137)
- 多语言对齐层 (V138)
- 隐私保护检索 (V139)
- 可解释性仪表盘 (V140)
- 强化学习优化器 v2 (V141)
- 增量图更新机制 (V142)
- 边缘部署支持 (V143)
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

router = APIRouter(prefix="/api/innovation", tags=["innovation"])


# ==================== Request/Response Models ====================

class MultimodalFusionRequest(BaseModel):
    """多模态融合请求"""
    question: str = Field(..., description="用户问题")
    evidence_chunks: List[Dict[str, Any]] = Field(..., description="证据块列表")
    enable_vision: bool = Field(default=False, description="是否启用图像分析")
    enable_code_analysis: bool = Field(default=True, description="是否启用代码分析")
    enable_table_extraction: bool = Field(default=True, description="是否启用表格提取")
    enable_formula_parsing: bool = Field(default=True, description="是否启用公式解析")


class TemporalReasoningRequest(BaseModel):
    """时序推理请求"""
    question: str = Field(..., description="用户问题")
    evidence_chunks: List[Dict[str, Any]] = Field(..., description="证据块列表")
    enable_causal_inference: bool = Field(default=True, description="是否启用因果推断")
    enable_trend_analysis: bool = Field(default=True, description="是否启用趋势分析")
    max_timeline_events: int = Field(default=50, description="最大事件数量")


class CrossLingualRequest(BaseModel):
    """多语言对齐请求"""
    question: str = Field(..., description="用户问题")
    evidence_chunks: List[Dict[str, Any]] = Field(..., description="证据块列表")
    primary_language: Optional[str] = Field(default=None, description="主要语言(zh/en)")
    enable_entity_extraction: bool = Field(default=True, description="是否启用实体提取")
    enable_translation: bool = Field(default=True, description="是否启用翻译")


class PrivacyRetrievalRequest(BaseModel):
    """隐私保护检索请求"""
    question: str = Field(..., description="用户问题")
    evidence_chunks: List[Dict[str, Any]] = Field(..., description="证据块列表")
    user_info: Dict[str, Any] = Field(..., description="用户信息 {user_id, roles}")
    enable_detection: bool = Field(default=True, description="是否启用敏感检测")
    enable_anonymization: bool = Field(default=True, description="是否启用匿名化")
    anonymization_strategy: str = Field(default="full_mask", description="脱敏策略")


class ExplainabilityRequest(BaseModel):
    """可解释性请求"""
    query: str = Field(..., description="用户查询")
    answer: str = Field(..., description="生成的答案")
    final_confidence: float = Field(..., ge=0, le=1, description="最终置信度")
    evidence_chunks: List[Dict[str, Any]] = Field(..., description="证据块列表")
    module_executions: Optional[List[Dict[str, Any]]] = Field(default=None, description="模块执行记录")


class RLOptimizerRequest(BaseModel):
    """RL优化器请求"""
    question: str = Field(..., description="用户问题")
    evidence_chunks: List[Dict[str, Any]] = Field(..., description="证据块列表")
    history_performance: Optional[List[float]] = Field(default=None, description="历史性能记录")
    enable_learning: bool = Field(default=True, description="是否启用学习")
    exploration_rate: float = Field(default=0.1, ge=0, le=1, description="探索率")


class IncrementalUpdateRequest(BaseModel):
    """增量更新请求"""
    current_entities: List[Dict[str, Any]] = Field(..., description="当前实体列表")
    current_relations: List[Dict[str, Any]] = Field(..., description="当前关系列表")
    new_documents: List[Dict[str, Any]] = Field(..., description="新文档列表")
    deleted_document_ids: Optional[List[str]] = Field(default=None, description="要删除的文档ID")
    auto_apply: bool = Field(default=False, description="是否自动应用更新")


class EdgeDeploymentRequest(BaseModel):
    """边缘部署请求"""
    custom_config: Optional[Dict[str, Any]] = Field(default=None, description="自定义配置")
    prepare_offline: bool = Field(default=False, description="是否准备离线数据")
    essential_knowledge: Optional[List[Dict[str, Any]]] = Field(default=None, description="必要知识数据")


# ==================== V136: 多模态知识融合 ====================

@router.post("/v136/multimodal-fusion", summary="多模态知识融合")
async def multimodal_fusion(request: MultimodalFusionRequest):
    """
    执行多模态知识融合分析
    
    - 提取表格、公式、代码等多模态信息
    - 构建跨模态关系和融合证据
    - 返回结构化的多模态分析结果
    """
    try:
        from api.features.chats.multimodal_knowledge_fusion import build_multimodal_evidence_fusion
        
        result = build_multimodal_evidence_fusion(
            question=request.question,
            evidence_chunks=[{"text": c.get("text", ""), "source_id": c.get("source_id", "")} 
                            for c in request.evidence_chunks],
            enable_vision=request.enable_vision,
            enable_code_analysis=request.enable_code_analysis,
            enable_table_extraction=request.enable_table_extraction,
            enable_formula_parsing=request.enable_formula_parsing
        )
        
        return {
            "success": True,
            "module": "V136_Multimodal_Knowledge_Fusion",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v136/table-extract", summary="表格提取测试")
async def table_extract_demo(text: str = Query(..., description="包含表格的文本")):
    """演示表格提取功能"""
    try:
        from api.features.chats.multimodal_knowledge_fusion import TableStructureExtractor
        
        result = TableStructureExtractor.extract(text)
        return {
            "success": True,
            "input_text": text[:200],
            "extracted_data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== V137: 时序推理增强 ====================

@router.post("/v137/temporal-reasoning", summary="时序推理增强")
async def temporal_reasoning(request: TemporalReasoningRequest):
    """
    执行时序推理分析
    
    - 从文本中提取时间表达式和事件
    - 构建时间线和因果关系
    - 检测趋势变化
    """
    try:
        from api.features.chats.temporal_reasoning_enhancement import build_temporal_reasoning_enhancement
        
        result = build_temporal_reasoning_enhancement(
            question=request.question,
            evidence_chunks=[{"text": c.get("text", ""), "source_id": c.get("source_id", "")} 
                            for c in request.evidence_chunks],
            enable_causal_inference=request.enable_causal_inference,
            enable_trend_analysis=request.enable_trend_analysis,
            max_timeline_events=request.max_timeline_events
        )
        
        return {
            "success": True,
            "module": "V137_Temporal_Reasoning_Enhancement",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== V138: 多语言对齐层 ====================

@router.post("/v138/cross-lingual", summary="多语言对齐")
async def cross_lingual_alignment(request: CrossLingualRequest):
    """
    执行跨语言对齐分析
    
    - 中英文实体识别与标准化
    - 翻译候选生成
    - 跨语言语义对齐
    """
    try:
        from api.features.chats.cross_lingual_alignment import build_cross_lingual_alignment
        
        result = build_cross_lingual_alignment(
            question=request.question,
            evidence_chunks=[{"text": c.get("text", ""), "source_id": c.get("source_id", "")} 
                            for c in request.evidence_chunks],
            primary_language=request.primary_language,
            enable_entity_extraction=request.enable_entity_extraction,
            enable_translation=request.enable_translation
        )
        
        return {
            "success": True,
            "module": "V138_Cross_Lingual_Alignment",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== V139: 隐私保护检索 ====================

@router.post("/v139/privacy-retrieval", summary="隐私保护检索")
async def privacy_retrieval(request: PrivacyRetrievalRequest):
    """
    执行隐私保护检索
    
    - PII 敏感信息检测
    - 文本脱敏处理
    - 访问控制检查
    - 审计日志记录
    """
    try:
        from api.features.chats.privacy_preserving_retrieval import build_privacy_preserving_retrieval
        
        result = build_privacy_preserving_retrieval(
            question=request.question,
            evidence_chunks=[{"text": c.get("text", ""), "source_id": c.get("source_id", "")} 
                            for c in request.evidence_chunks],
            user_info=request.user_info,
            enable_detection=request.enable_detection,
            enable_anonymization=request.enable_anonymization,
            anonymization_strategy=request.anonymization_strategy
        )
        
        return {
            "success": True,
            "module": "V139_Privacy_Preserving_Retrieval",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v139/detect-pii", summary="PII检测")
async def detect_pii(text: str = Body(..., embed=True, description="要检测的文本")):
    """快速检测文本中的个人身份信息"""
    try:
        from api.features.chats.privacy_preserving_retrieval import SensitiveDataDetector
        
        entities = SensitiveDataDetector.detect(text)
        
        return {
            "success": True,
            "input_length": len(text),
            "entities_found": len(entities),
            "entities": [e.to_dict() for e in entities]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== V140: 可解释性仪表盘 ====================

@router.post("/v140/explainability-dashboard", summary="可解释性仪表盘")
async def explainability_dashboard(request: ExplainabilityRequest):
    """
    生成可解释性仪表盘数据
    
    - 证据树构建
    - 推理路径追踪
    - 置信度分解
    - 模块贡献分析
    """
    try:
        from api.features.chats.explainability_dashboard import build_explainability_dashboard_data
        
        result = build_explainability_dashboard_data(
            query=request.query,
            answer=request.answer,
            final_confidence=request.final_confidence,
            evidence_chunks=request.evidence_chunks,
            module_executions=request.module_executions
        )
        
        return {
            "success": True,
            "module": "V140_Explainability_Dashboard",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== V141: 强化学习优化器 v2 ====================

@router.post("/v141/rl-optimizer", summary="强化学习优化器")
async def rl_optimizer(request: RLOptimizerRequest):
    """
    执行 RL 优化分析
    
    - 当前状态评估
    - 最优动作选择
    - 奖励预估
    - 优化建议生成
    """
    try:
        from api.features.chats.reinforcement_learning_optimizer_v2 import build_rl_optimizer_v2
        
        result = build_rl_optimizer_v2(
            question=request.question,
            evidence_chunks=request.evidence_chunks,
            history_performance=request.history_performance,
            enable_learning=request.enable_learning,
            exploration_rate=request.exploration_rate
        )
        
        return {
            "success": True,
            "module": "V141_RL_Optimizer_v2",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== V142: 增量图更新机制 ====================

@router.post("/v142/incremental-update", summary="增量图更新")
async def incremental_update(request: IncrementalUpdateRequest):
    """
    执行增量图更新分析
    
    - 变更检测
    - 影响范围评估
    - 更新计划生成
    - 版本快照预览
    """
    try:
        from api.features.chats.incremental_graph_update import build_incremental_update_pipeline
        
        result = build_incremental_update_pipeline(
            current_entities=request.current_entities,
            current_relations=request.current_relations,
            new_documents=request.new_documents,
            deleted_document_ids=request.deleted_document_ids,
            update_config={"auto_apply": request.auto_apply}
        )
        
        return {
            "success": True,
            "module": "V142_Incremental_Graph_Update",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== V143: 边缘部署支持 ====================

@router.post("/v143/edge-deployment", summary="边缘部署支持")
async def edge_deployment(request: EdgeDeploymentRequest):
    """
    执行边缘部署分析
    
    - 系统资源检测
    - 最优配置生成
    - 缓存管理初始化
    - 就绪度评估
    """
    try:
        from api.features.chats.edge_deployment_support import build_edge_deployment_support
        
        result = build_edge_deployment_support(
            custom_config=request.custom_config,
            prepare_offline=request.prepare_offline,
            essential_knowledge=request.essential_knowledge
        )
        
        return {
            "success": True,
            "module": "V143_Edge_Deployment_Support",
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v143/system-resources", summary="获取系统资源")
async def get_system_resources():
    """获取当前系统资源信息"""
    try:
        from api.features.chats.edge_deployment_support import EnvironmentDetector
        
        resources = EnvironmentDetector.detect_system_resources()
        platform = EnvironmentDetector.get_platform_info()
        tier = EnvironmentDetector.detect_device_tier(resources)
        
        return {
            "success": True,
            "resources": resources.to_dict(),
            "platform": platform,
            "device_tier": tier.value,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 综合接口 ====================

@router.get("/", summary="获取所有创新模块概览")
async def get_innovation_modules_overview():
    """获取 V136-V143 所有创新模块的概览信息"""
    modules = [
        {
            "id": "V136",
            "name": "多模态知识融合引擎",
            "name_en": "Multimodal Knowledge Fusion",
            "description": "将图像、表格、公式、代码作为一等公民融入知识图谱推理过程",
            "endpoint": "/api/innovation/v136/multimodal-fusion",
            "capabilities": ["表格提取", "公式解析", "代码分析", "跨模态融合"],
            "status": "active"
        },
        {
            "id": "V137",
            "name": "时序推理增强",
            "name_en": "Temporal Reasoning Enhancement",
            "description": "支持时间维度的事件因果链推理，包括时间线构建与趋势预测",
            "endpoint": "/api/innovation/v137/temporal-reasoning",
            "capabilities": ["时间表达式提取", "事件排序", "因果推断", "趋势分析"],
            "status": "active"
        },
        {
            "id": "V138",
            "name": "多语言对齐层",
            "name_en": "Cross-Lingual Alignment Layer",
            "description": "支持中英双语实体的深度语义对齐，包括翻译验证与文化感知",
            "endpoint": "/api/innovation/v138/cross-lingual",
            "capabilities": ["双语实体识别", "术语翻译", "语义对齐", "文化感知"],
            "status": "active"
        },
        {
            "id": "V139",
            "name": "隐私保护检索",
            "name_en": "Privacy-Preserving Retrieval",
            "description": "实现联邦学习+差分隐私的知识图谱查询，包括PII检测与脱敏",
            "endpoint": "/api/innovation/v139/privacy-retrieval",
            "capabilities": ["PII检测", "数据脱敏", "访问控制", "审计日志"],
            "status": "active"
        },
        {
            "id": "V140",
            "name": "可解释性仪表盘",
            "name_en": "Explainability Dashboard",
            "description": "为前端提供完整的推理过程可视化数据，包括证据溯源与置信度分解",
            "endpoint": "/api/innovation/v140/explainability-dashboard",
            "capabilities": ["证据树", "推理路径", "置信度分解", "模块贡献"],
            "status": "active"
        },
        {
            "id": "V141",
            "name": "强化学习优化器 v2",
            "name_en": "RL Optimizer v2",
            "description": "基于PPO的端到端优化框架，动态调整检索策略与答案生成参数",
            "endpoint": "/api/innovation/v141/rl-optimizer",
            "capabilities": ["策略网络", "多目标奖励", "在线学习", "动作选择"],
            "status": "active"
        },
        {
            "id": "V142",
            "name": "增量图更新机制",
            "name_en": "Incremental Graph Update",
            "description": "支持知识图谱的增量式更新，无需全量重建，包括版本管理与回滚",
            "endpoint": "/api/innovation/v142/incremental-update",
            "capabilities": ["变更检测", "影响评估", "版本管理", "一致性维护"],
            "status": "active"
        },
        {
            "id": "V143",
            "name": "边缘部署支持",
            "name_en": "Edge Deployment Support",
            "description": "支持轻量级模型部署到边缘设备，包括模型量化压缩与离线模式",
            "endpoint": "/api/innovation/v143/edge-deployment",
            "capabilities": ["环境检测", "模型压缩", "缓存优化", "离线模式"],
            "status": "active"
        }
    ]
    
    return {
        "success": True,
        "total_modules": len(modules),
        "version_range": "V136-V143",
        "modules": modules
    }


@router.post("/batch-process", summary="批量处理多个模块")
async def batch_process(
    question: str = Body(..., description="用户问题"),
    evidence_chunks: List[Dict[str, Any]] = Body(..., description="证据块"),
    modules: Optional[List[str]] = Body(default=None, description="要启用的模块ID列表")
):
    """
    批量调用多个创新模块进行处理
    
    可以一次性调用多个模块，返回综合结果
    """
    if modules is None:
        modules = ["V136", "V137", "V138", "V140"]
    
    results = {}
    
    # V136: 多模态融合
    if "V136" in modules:
        try:
            from api.features.chats.multimodal_knowledge_fusion import build_multimodal_evidence_fusion
            results["V136"] = build_multimodal_evidence_fusion(
                question=question,
                evidence_chunks=evidence_chunks
            )
        except Exception as e:
            results["V136"] = {"error": str(e)}
    
    # V137: 时序推理
    if "V137" in modules:
        try:
            from api.features.chats.temporal_reasoning_enhancement import build_temporal_reasoning_enhancement
            results["V137"] = build_temporal_reasoning_enhancement(
                question=question,
                evidence_chunks=evidence_chunks
            )
        except Exception as e:
            results["V137"] = {"error": str(e)}
    
    # V138: 多语言对齐
    if "V138" in modules:
        try:
            from api.features.chats.cross_lingual_alignment import build_cross_lingual_alignment
            results["V138"] = build_cross_lingual_alignment(
                question=question,
                evidence_chunks=evidence_chunks
            )
        except Exception as e:
            results["V138"] = {"error": str(e)}
    
    # V140: 可解释性
    if "V140" in modules:
        try:
            from api.features.chats.explainability_dashboard import build_explainability_dashboard_data
            results["V140"] = build_explainability_dashboard_data(
                query=question,
                answer="",  # 需要实际答案
                final_confidence=0.85,
                evidence_chunks=evidence_chunks
            )
        except Exception as e:
            results["V140"] = {"error": str(e)}
    
    return {
        "success": True,
        "processed_modules": list(results.keys()),
        "question": question,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
