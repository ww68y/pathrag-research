import json
import asyncio
import numpy as np
from tqdm import tqdm
from collections import Counter
import os
import re
import json
import time
from .PathRAG import PathRAG, QueryParam
from .llm import gpt_4o_mini_complete
import asyncio
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
from tqdm import tqdm
import time
import json
from wordfreq import word_frequency
from .utils import (
    logger,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    split_string_by_multi_markers,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


# 全局收集池
pending_entities_vdb_upsert = {}
pending_relationships_vdb_upsert = {}
async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.info(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary
async def merge_node(
    nodes_data: list[dict],  # 要合并的节点们
    replace_name,
    knowledge_graph_inst,
    entities_vdb,
    relationships_vdb,
    global_config,
    replace,
):
    if not nodes_data:
        return None

    merged_node_name = GRAPH_FIELD_SEP.join(nodes_data)
    if replace:
        merged_node_name = replace_name 
    # ======== 合并节点属性 ========
    all_entity_types = []
    all_source_ids = []
    all_descriptions = []

    for node in nodes_data:
        temp_node =  await knowledge_graph_inst.get_node(node)
        all_entity_types.append(temp_node["entity_type"])
        all_source_ids.extend(
            split_string_by_multi_markers(temp_node["source_id"], [GRAPH_FIELD_SEP])
        )
        all_descriptions.append(temp_node["description"])

    # 选择出现次数最多的 entity_type
    entity_type = sorted(
        Counter(all_entity_types).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    description = GRAPH_FIELD_SEP.join(
        sorted(set(all_descriptions))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set(all_source_ids)
    )

    description = await _handle_entity_relation_summary(
        merged_node_name, description, global_config
    )

    merged_node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    
    await knowledge_graph_inst.upsert_node(merged_node_name, merged_node_data)
    # print("node:",merged_node_name, merged_node_data)

    all_edges = []
    for node in nodes_data:
        edges = []
        temp_edges = await knowledge_graph_inst.get_node_edges(node) # 无向图统一叫 get_edges
        for e in temp_edges:
            sorted_edge = tuple(sorted(e))
            edges.append(sorted_edge)
        # print(edges)
        if edges:
            all_edges_pack = await asyncio.gather(
                *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in edges]
            )
            # print(all_edges_pack)
            all_edges_data = [
                {"source": k[0], "target":k[1], **v}
                for k, v in zip(edges, all_edges_pack)
                if v is not None
            ]
            # print(all_edges_data)
            all_edges.extend(all_edges_data)
    
    merged_from_names = set(node for node in nodes_data)

    edge_data_map = {}
    for edge in all_edges:
        node1 = edge["source"]
        node2 = edge["target"]

        # 自己指向自己的边忽略
        if node1 in merged_from_names and node2 in merged_from_names:
            continue
        other_node = node2 if node1 in merged_from_names else node1

        node_pair = tuple(sorted([merged_node_name, other_node]))
        if node_pair not in edge_data_map:
            edge_data_map[node_pair] = []

        edge_data_map[node_pair].append(edge)
    for node_pair, edges_data in edge_data_map.items():
            # 聚合各字段
        # print(node_pair)
        # print(edges_data)
        weight = sum([dp["weight"] for dp in edges_data]) 
        description = GRAPH_FIELD_SEP.join(
            sorted(set([dp["description"] for dp in edges_data]))
        )
        keywords = GRAPH_FIELD_SEP.join(
            sorted(set([dp["keywords"] for dp in edges_data]))
        )
        source_id = GRAPH_FIELD_SEP.join(
            set([dp["source_id"] for dp in edges_data])
        ) 
        edge_data = {
            "weight": weight,
            "description": description,
            "keywords": keywords,
            "source_id": source_id,
        }
        # print(edge_data)
        node1, node2 = node_pair
    
        await knowledge_graph_inst.upsert_edge(node1, node2, edge_data)
    # ======== 删除原节点 ========
    for node in nodes_data:
        if node!= replace_name:
            print(node, replace_name)
            await knowledge_graph_inst.delete_node(node)
    
    return knowledge_graph_inst

async def merge(file_path, rag, stname:'all-mpnet-base-v2'):
    file_to_delete1 = os.path.join(file_path, "vdb_entities.json")
    file_to_delete2 = os.path.join(file_path, "vdb_relationships.json")
    if os.path.exists(file_to_delete1):
        os.remove(file_to_delete1)
        print(f"{file_to_delete1} 已成功删除。")
    else:
        print(f"{file_to_delete1} 不存在。")
    
    if os.path.exists(file_to_delete2):
        os.remove(file_to_delete2)
        print(f"{file_to_delete2} 已成功删除。")
    else:
        print(f"{file_to_delete2} 不存在。")
    knowledge_graph_inst = rag.chunk_entity_relation_graph
    global_config = {
        "llm_model_func": gpt_4o_mini_complete,
        "llm_model_max_token_size": 32768,
        "tiktoken_model_name": "gpt-4o-mini",
        "entity_summary_to_max_tokens": 500,
    }
    entities_vdb=rag.entities_vdb
    relationships_vdb=rag.relationships_vdb

    node_list = list(await knowledge_graph_inst.nodes())
    edge_list = list(await knowledge_graph_inst.edges())
    words = node_list




    model = SentenceTransformer(stname)
    embeddings = model.encode(words, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # 使用 DBSCAN 聚类
    dbscan = DBSCAN(eps=0.12, min_samples=1, metric='cosine', n_jobs=-1)  
    labels = dbscan.fit_predict(embeddings)

    # 将聚类结果输出
    clusters = {}
    for idx, label in enumerate(labels):
        if label != -1:  # -1 是噪声点
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(words[idx])
    for cluster_id, cluster_words in clusters.items():
        if len(cluster_words) > 1:
            scores = [word_frequency(word.lower(), 'en') for word in cluster_words]
            best_word = cluster_words[scores.index(max(scores))]
            replace_name = f"{best_word}"
            print(replace_name)
            replace = True
            knowledge_graph_inst = await merge_node(cluster_words, replace_name, knowledge_graph_inst, entities_vdb, relationships_vdb, global_config, replace)
    # node_list = list(await knowledge_graph_inst.nodes())
    # edge_list = list(await knowledge_graph_inst.edges())
    node_list = list(await knowledge_graph_inst.nodes())
    edge_list = list(await knowledge_graph_inst.edges())
    for node in node_list:
        node_data = await knowledge_graph_inst.get_node(node)
        # print(node, node_data)
        data_for_vdb_entities = {
            compute_mdhash_id(node, prefix="ent-"): {
                "content": node + node_data["description"],
                "entity_name": node,
            }
        }
        # print(data_for_vdb_entities)
        pending_entities_vdb_upsert.update(data_for_vdb_entities)
    for edge in edge_list:
        edge_data = await knowledge_graph_inst.get_edge(edge[0], edge[1])
        # print(edge_data)
        data_for_vdb_relationships = {
            compute_mdhash_id(edge[0] + edge[1], prefix="rel-"): {
                "src_id": edge[0],
                "tgt_id": edge[1],
                "content": edge_data["keywords"] + edge[0] + edge[1] + edge_data["description"],
            }
        }
        pending_relationships_vdb_upsert.update(data_for_vdb_relationships)

    print(f"准备写入 {len(pending_entities_vdb_upsert)} 个节点到 VDB...")
    await entities_vdb.upsert(pending_entities_vdb_upsert)

    print(f"准备写入 {len(pending_relationships_vdb_upsert)} 条关系到 VDB...")
    await relationships_vdb.upsert(pending_relationships_vdb_upsert)
    await rag._insert_done()



if __name__ == "__main__":
    # domain = ["agriculture", "history", "legal","legal",]
    for subdomain in ["mix"]:
        WORKING_DIR = f"./dickens_{subdomain}_v1"
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)
        rag = PathRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,
        )
        asyncio.run(merge(file_path=f"./dickens_{subdomain}_v1", rag=rag))

# python -m lightrag.merge