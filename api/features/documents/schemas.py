from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str
    content_type: str

class DocumentCreate(DocumentBase):
    file_size: int
    file_path: str

class DocumentResponse(DocumentBase):
    id: int
    user_id: int
    file_size: int
    uploaded_at: datetime
    status: Optional[str] = None
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    model_config = {
        "from_attributes": True
    }

class DocumentList(BaseModel):
    documents: List[DocumentResponse]

    model_config = {
        "from_attributes": True
    }

class UploadStatus(BaseModel):
    filename: str
    content_type: str
    progress: float  # 0-100 percentage
    status: str  # "processing", "completed", "failed"
    message: Optional[str] = None
    document_id: Optional[int] = None

# Neo4j Entity Schemas
class EntityNode(BaseModel):
    id: str
    entity_type: Optional[str] = None
    description: Optional[str] = None
    additional_descriptions: Optional[List[str]] = None
    source_id: Optional[str] = None
    source_file: Optional[str] = None
    node_id: Optional[str] = None

class EntitySearchRequest(BaseModel):
    entity_name: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: Optional[str] = "neo4j"

class EntitySearchResponse(BaseModel):
    success: bool
    message: str
    entity: Optional[EntityNode] = None

class SubgraphRequest(BaseModel):
    entity_list: List[str]
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: Optional[str] = "neo4j"

class RelationshipEdge(BaseModel):
    source: str
    target: str
    weight: Optional[float] = None
    description: Optional[str] = None
    keywords: Optional[str] = None
    source_id: Optional[str] = None
    source_file: Optional[str] = None

class SubgraphResponse(BaseModel):
    success: bool
    message: str
    nodes: List[EntityNode]
    relationships: List[RelationshipEdge]
    nodes_count: int
    relationships_count: int
