import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from api.features.rag_manager import get_rag_instance
from models.database import User
from api.auth.jwt_handler import get_current_active_user
from .schemas import Graph, GraphQuery, Node, Edge

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_TRACE_FILE = "mse_pathrag_sage_v8_uncertainty_hyde_biology_20.jsonl"

router = APIRouter(
    prefix="/knowledge-graph",
    tags=["Knowledge Graph"],
    dependencies=[Depends(get_current_active_user)]
)


def _results_jsonl_files() -> list[str]:
    results_dir = ROOT_DIR / "results"
    if not results_dir.exists():
        return []
    return sorted(path.name for path in results_dir.glob("*.jsonl") if path.is_file())


def _safe_results_file(file_name: str) -> Path:
    safe_name = Path(file_name).name
    if safe_name != file_name or not safe_name.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Trace file must be a JSONL filename from the results directory.")
    path = ROOT_DIR / "results" / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Trace file not found: {safe_name}")
    return path


def _load_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _compact_trace(row: dict) -> dict:
    trace = row.get("evidence_trace") or {}
    return {
        "id": row.get("id"),
        "line_no": row.get("line_no"),
        "dataset": row.get("dataset"),
        "label": row.get("label"),
        "method": row.get("method"),
        "question": row.get("question"),
        "reference_answers": row.get("reference_answers", []),
        "prediction": row.get("prediction", ""),
        "original_prediction": row.get("original_prediction", ""),
        "graph_prediction_before_router": row.get("graph_prediction_before_router", ""),
        "prediction_before_candidate_router": row.get("prediction_before_candidate_router", ""),
        "prediction_before_rescue": row.get("prediction_before_rescue", ""),
        "prediction_before_compression": row.get("prediction_before_compression", ""),
        "prediction_before_answer_contract_rescue": row.get("prediction_before_answer_contract_rescue", ""),
        "prediction_before_text_first_graph_rescue": row.get("prediction_before_text_first_graph_rescue", ""),
        "prediction_before_budget_meta_route": row.get("prediction_before_budget_meta_route", ""),
        "prediction_before_learned_graph_accept": row.get("prediction_before_learned_graph_accept", ""),
        "prediction_before_claim_guard": row.get("prediction_before_claim_guard", ""),
        "prediction_before_contract_completion": row.get("prediction_before_contract_completion", ""),
        "prediction_before_postrepair_consensus": row.get("prediction_before_postrepair_consensus", ""),
        "prediction_before_selective_consensus": row.get("prediction_before_selective_consensus", ""),
        "suppressed_contract_completion_prediction": row.get("suppressed_contract_completion_prediction", ""),
        "rejected_evidence_delta_prediction": row.get("rejected_evidence_delta_prediction", ""),
        "prediction_before_candidate_arbitration": row.get("prediction_before_candidate_arbitration", ""),
        "prediction_before_context_graph_memory": row.get("prediction_before_context_graph_memory", ""),
        "prediction_before_v39_pregraph_plan": row.get("prediction_before_v39_pregraph_plan", ""),
        "prediction_before_v41_pregraph_plan": row.get("prediction_before_v41_pregraph_plan", ""),
        "prediction_before_evidence_calibrated_lazy_agent": row.get("prediction_before_evidence_calibrated_lazy_agent", ""),
        "prediction_before_text_evidence_expansion_fallback": row.get(
            "prediction_before_text_evidence_expansion_fallback", ""
        ),
        "prediction_before_evidence_mismatch_repair": row.get("prediction_before_evidence_mismatch_repair", ""),
        "prediction_before_conservative_repair_compression": row.get(
            "prediction_before_conservative_repair_compression", ""
        ),
        "prediction_before_question_anchored_repair": row.get("prediction_before_question_anchored_repair", ""),
        "prediction_before_question_chunk_completion": row.get("prediction_before_question_chunk_completion", ""),
        "prediction_before_question_completion_compression": row.get(
            "prediction_before_question_completion_compression", ""
        ),
        "prediction_before_direct_fact_compression": row.get("prediction_before_direct_fact_compression", ""),
        "prediction_before_self_corrected_fact_compression": row.get(
            "prediction_before_self_corrected_fact_compression", ""
        ),
        "prediction_before_query_expanded_refusal_repair": row.get(
            "prediction_before_query_expanded_refusal_repair", ""
        ),
        "prediction_before_substance_refusal_repair": row.get("prediction_before_substance_refusal_repair", ""),
        "prediction_before_query_expanded_contract_gap_repair": row.get(
            "prediction_before_query_expanded_contract_gap_repair", ""
        ),
        "prediction_before_supported_facet_guard": row.get("prediction_before_supported_facet_guard", ""),
        "prediction_before_supported_facet_fusion": row.get("prediction_before_supported_facet_fusion", ""),
        "prediction_before_facet_balanced_compaction": row.get("prediction_before_facet_balanced_compaction", ""),
        "prediction_before_residual_hardcase_editor": row.get("prediction_before_residual_hardcase_editor", ""),
        "prediction_before_evidence_risk_gate": row.get("prediction_before_evidence_risk_gate", ""),
        "prediction_before_dual_objective_gate": row.get("prediction_before_dual_objective_gate", ""),
        "prediction_before_kg_rag_agent": row.get("prediction_before_kg_rag_agent", ""),
        "rejected_graph_rescue_prediction": row.get("rejected_graph_rescue_prediction", ""),
        "rejected_learned_graph_prediction": row.get("rejected_learned_graph_prediction", ""),
        "suppressed_graph_prediction_by_context_budget": row.get("suppressed_graph_prediction_by_context_budget", ""),
        "rejected_contract_completion_prediction": row.get("rejected_contract_completion_prediction", ""),
        "rejected_evidence_mismatch_repair_prediction": row.get("rejected_evidence_mismatch_repair_prediction", ""),
        "rejected_conservative_repair_compression": row.get("rejected_conservative_repair_compression", ""),
        "rejected_question_chunk_completion_prediction": row.get("rejected_question_chunk_completion_prediction", ""),
        "rejected_question_completion_compression": row.get("rejected_question_completion_compression", ""),
        "rejected_direct_fact_compression": row.get("rejected_direct_fact_compression", ""),
        "rejected_self_corrected_fact_compression": row.get("rejected_self_corrected_fact_compression", ""),
        "rejected_query_expanded_refusal_repair": row.get("rejected_query_expanded_refusal_repair", ""),
        "rejected_substance_refusal_repair": row.get("rejected_substance_refusal_repair", ""),
        "rejected_query_expanded_contract_gap_repair": row.get("rejected_query_expanded_contract_gap_repair", ""),
        "rejected_supported_facet_fusion": row.get("rejected_supported_facet_fusion", ""),
        "rejected_facet_balanced_compaction": row.get("rejected_facet_balanced_compaction", ""),
        "rejected_evidence_risk_gate_prediction": row.get("rejected_evidence_risk_gate_prediction", ""),
        "rejected_dual_objective_gate_prediction": row.get("rejected_dual_objective_gate_prediction", ""),
        "uncertainty_fallback": row.get("uncertainty_fallback", {}),
        "query_router": row.get("query_router", {}),
        "legal_error_router": row.get("legal_error_router", {}),
        "cost_aware_prerouter": row.get("cost_aware_prerouter", {}),
        "answer_contract_agent": row.get("answer_contract_agent", {}),
        "text_first_contract_agent": row.get("text_first_contract_agent", {}),
        "budget_meta_agent": row.get("budget_meta_agent", {}),
        "learned_dual_gate": row.get("learned_dual_gate", {}),
        "context_budget_agent": row.get("context_budget_agent", {}),
        "claim_evidence_agent": row.get("claim_evidence_agent", {}),
        "contract_completion_agent": row.get("contract_completion_agent", {}),
        "post_repair_consensus_agent": row.get("post_repair_consensus_agent", {}),
        "selective_consensus_router": row.get("selective_consensus_router", {}),
        "learned_contract_gap_router": row.get("learned_contract_gap_router", {}),
        "evidence_delta_guard": row.get("evidence_delta_guard", {}),
        "evidence_candidate_arbiter": row.get("evidence_candidate_arbiter", {}),
        "context_graph_memory_agent": row.get("context_graph_memory_agent", {}),
        "v39_pregraph_plan": row.get("v39_pregraph_plan", {}),
        "v41_pregraph_plan": row.get("v41_pregraph_plan", {}),
        "evidence_calibrated_lazy_agent": row.get("evidence_calibrated_lazy_agent", {}),
        "text_evidence_expansion_fallback": row.get("text_evidence_expansion_fallback", {}),
        "evidence_mismatch_repair": row.get("evidence_mismatch_repair", {}),
        "conservative_repair_compression": row.get("conservative_repair_compression", {}),
        "question_anchored_repair": row.get("question_anchored_repair", {}),
        "question_chunk_completion": row.get("question_chunk_completion", {}),
        "question_completion_compression": row.get("question_completion_compression", {}),
        "direct_fact_compression": row.get("direct_fact_compression", {}),
        "self_corrected_fact_compression": row.get("self_corrected_fact_compression", {}),
        "query_expanded_refusal_repair": row.get("query_expanded_refusal_repair", {}),
        "substance_refusal_repair": row.get("substance_refusal_repair", {}),
        "query_expanded_contract_gap_repair": row.get("query_expanded_contract_gap_repair", {}),
        "supported_facet_guard": row.get("supported_facet_guard", {}),
        "supported_facet_fusion": row.get("supported_facet_fusion", {}),
        "facet_balanced_compaction": row.get("facet_balanced_compaction", {}),
        "residual_hardcase_editor": row.get("residual_hardcase_editor", {}),
        "evidence_risk_gate": row.get("evidence_risk_gate", {}),
        "dual_objective_gate": row.get("dual_objective_gate", {}),
        "kg_rag_agent": row.get("kg_rag_agent", {}),
        "rescue_compression": row.get("rescue_compression", {}),
        "query_seconds": row.get("query_seconds"),
        "graph_nodes": row.get("graph_nodes"),
        "graph_edges": row.get("graph_edges"),
        "working_dir": row.get("working_dir"),
        "mse_stats": row.get("mse_stats", {}),
        "query_analysis": trace.get("query_analysis", {}),
        "policy": trace.get("policy", {}),
        "selection_summary": trace.get("selection_summary", {}),
        "sufficiency": trace.get("sufficiency", {}),
        "answer_aspects": trace.get("answer_aspects", row.get("answer_aspects", [])),
        "aspect_preservation": trace.get("aspect_preservation", {}),
        "selected_edges": trace.get("selected_edges", []),
        "top_candidates": trace.get("top_candidates", []),
        "filtered_counterfactual_edges": trace.get("filtered_counterfactual_edges", []),
        "supporting_chunks": trace.get("supporting_chunks", []),
        "prompt_stats": trace.get("prompt_stats", {}),
    }


def _has_trace_data(row: dict) -> bool:
    return bool(
        row.get("evidence_trace")
        or row.get("answer_contract_agent")
        or row.get("text_first_contract_agent")
        or row.get("budget_meta_agent")
        or row.get("learned_dual_gate")
        or row.get("context_budget_agent")
        or row.get("claim_evidence_agent")
        or row.get("contract_completion_agent")
        or row.get("post_repair_consensus_agent")
        or row.get("selective_consensus_router")
        or row.get("learned_contract_gap_router")
        or row.get("evidence_delta_guard")
        or row.get("evidence_candidate_arbiter")
        or row.get("context_graph_memory_agent")
        or row.get("v39_pregraph_plan")
        or row.get("v41_pregraph_plan")
        or row.get("evidence_calibrated_lazy_agent")
        or row.get("text_evidence_expansion_fallback")
        or row.get("evidence_mismatch_repair")
        or row.get("conservative_repair_compression")
        or row.get("question_anchored_repair")
        or row.get("question_chunk_completion")
        or row.get("question_completion_compression")
        or row.get("direct_fact_compression")
        or row.get("self_corrected_fact_compression")
        or row.get("query_expanded_refusal_repair")
        or row.get("substance_refusal_repair")
        or row.get("query_expanded_contract_gap_repair")
        or row.get("supported_facet_guard")
        or row.get("supported_facet_fusion")
        or row.get("facet_balanced_compaction")
        or row.get("residual_hardcase_editor")
        or row.get("evidence_risk_gate")
        or row.get("dual_objective_gate")
        or row.get("kg_rag_agent")
        or row.get("cost_aware_prerouter")
    )


@router.get("/evidence-traces")
async def get_evidence_traces(
    file_name: str = Query(DEFAULT_TRACE_FILE, description="JSONL file name under results/"),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
):
    path = _safe_results_file(file_name)
    rows = _load_jsonl(path, limit)
    traces = [_compact_trace(row) for row in rows if _has_trace_data(row)]
    return {
        "file": path.name,
        "count": len(traces),
        "available_files": _results_jsonl_files(),
        "traces": traces,
    }

@router.post("/query", response_model=Graph)
async def query_knowledge_graph(query: GraphQuery, current_user: User = Depends(get_current_active_user)):
    try:
        rag = get_rag_instance()
        # Get the knowledge graph from PathRAG
        knowledge_graph = rag.chunk_entity_relation_graph
        
        # Convert NetworkX graph to our schema
        nodes = []
        edges = []
        
        # Get all nodes and edges
        for node_id in knowledge_graph._graph.nodes():
            node_data = knowledge_graph._graph.nodes[node_id]
            nodes.append(
                Node(
                    id=node_id,
                    label=node_id,
                    type=node_data.get("entity_type", "Unknown"),
                    description=node_data.get("description", "")
                )
            )
        
        for source, target, data in knowledge_graph._graph.edges(data=True):
            edges.append(
                Edge(
                    source=source,
                    target=target,
                    label=data.get("description", ""),
                    weight=data.get("weight", 1.0)
                )
            )
        
        return Graph(nodes=nodes, edges=edges)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying knowledge graph: {str(e)}")

@router.get("/", response_model=Graph)
async def get_knowledge_graph(current_user: User = Depends(get_current_active_user)):
    try:
        rag = get_rag_instance()
        # Get the knowledge graph from PathRAG
        knowledge_graph = rag.chunk_entity_relation_graph
        
        # Convert NetworkX graph to our schema
        nodes = []
        edges = []
        
        # Get all nodes and edges
        for node_id in knowledge_graph._graph.nodes():
            node_data = knowledge_graph._graph.nodes[node_id]
            nodes.append(
                Node(
                    id=node_id,
                    label=node_id,
                    type=node_data.get("entity_type", "Unknown"),
                    description=node_data.get("description", "")
                )
            )
        
        for source, target, data in knowledge_graph._graph.edges(data=True):
            edges.append(
                Edge(
                    source=source,
                    target=target,
                    label=data.get("description", ""),
                    weight=data.get("weight", 1.0)
                )
            )
        
        return Graph(nodes=nodes, edges=edges)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting knowledge graph: {str(e)}")
