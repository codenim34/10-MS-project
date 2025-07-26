"""
REST API for Multilingual RAG System
Provides endpoints for document processing and question answering
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from multilingual_rag import MultilingualRAG
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG system
rag_system = MultilingualRAG()

# FastAPI app
app = FastAPI(
    title="Multilingual RAG API",
    description="Bengali-English Retrieval-Augmented Generation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query in Bengali or English")
    include_context: bool = Field(True, description="Include conversation context")
    top_k: int = Field(Config.TOP_K_CHUNKS, description="Number of chunks to retrieve")

class QueryResponse(BaseModel):
    query: str
    response: str
    language: str
    retrieved_chunks: List[Dict[str, Any]]
    conversation_context: str
    metadata: Dict[str, Any]
    timestamp: str

class SystemStats(BaseModel):
    vector_database: Dict[str, Any]
    conversation_memory: Dict[str, Any]
    llm_type: Optional[str]
    config: Dict[str, Any]

class SetupResponse(BaseModel):
    success: bool
    total_chunks: int
    processed_files: List[Dict[str, Any]]
    errors: List[str]
    database_stats: Dict[str, Any]

# API Endpoints

@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multilingual RAG API",
        "version": "1.0.0",
        "description": "Bengali-English Retrieval-Augmented Generation System",
        "endpoints": {
            "query": "/query",
            "setup": "/setup",
            "stats": "/stats",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", summary="Health check")
async def health_check():
    """Health check endpoint"""
    try:
        stats = rag_system.get_system_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database_status": "connected" if stats.get("vector_database") else "disconnected",
            "chunks_count": stats.get("vector_database", {}).get("total_chunks", 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"System unhealthy: {str(e)}")

@app.post("/query", response_model=QueryResponse, summary="Query the RAG system")
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a Bengali or English question
    
    - **query**: The question to ask (in Bengali or English)
    - **include_context**: Whether to include conversation context
    - **top_k**: Number of document chunks to retrieve
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Processing query: {request.query}")
        
        # Process query
        result = rag_system.query(
            user_query=request.query,
            include_context=request.include_context,
            top_k=request.top_k
        )
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/setup", response_model=SetupResponse, summary="Setup knowledge base")
async def setup_knowledge_base(background_tasks: BackgroundTasks):
    """
    Set up the knowledge base by processing PDF files
    This processes all PDF files in the configured directory
    """
    try:
        logger.info("Starting knowledge base setup")
        
        # Run setup
        result = rag_system.setup_knowledge_base()
        
        if not result["success"] and result["errors"]:
            raise HTTPException(
                status_code=500, 
                detail=f"Setup failed: {'; '.join(result['errors'])}"
            )
        
        return SetupResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge base setup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

@app.post("/upload-pdf", summary="Upload and process PDF file")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file to add to the knowledge base
    """
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file
        upload_path = os.path.join(Config.PDF_PATH, file.filename)
        os.makedirs(Config.PDF_PATH, exist_ok=True)
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded PDF: {file.filename}")
        
        # Process the uploaded file
        result = rag_system.setup_knowledge_base([upload_path])
        
        return {
            "filename": file.filename,
            "size": len(content),
            "processing_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/stats", response_model=SystemStats, summary="Get system statistics")
async def get_system_stats():
    """
    Get comprehensive system statistics including database and memory stats
    """
    try:
        stats = rag_system.get_system_stats()
        return SystemStats(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/clear-history", summary="Clear conversation history")
async def clear_conversation_history():
    """
    Clear all conversation history
    """
    try:
        rag_system.clear_conversation_history()
        return {"message": "Conversation history cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

@app.get("/export-conversations", summary="Export conversation history")
async def export_conversations(language: Optional[str] = None):
    """
    Export conversation history to JSON
    
    - **language**: Filter by language (bengali, english, mixed)
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversations_{timestamp}.json"
        output_path = os.path.join("data", filename)
        
        success = rag_system.export_conversation_history(output_path, language)
        
        if success:
            return {
                "message": "Conversation history exported successfully",
                "filename": filename,
                "path": output_path
            }
        else:
            raise HTTPException(status_code=500, detail="Export failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/test-queries", summary="Test with predefined queries")
async def test_predefined_queries():
    """
    Test the system with predefined Bengali queries
    """
    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]
    
    results = []
    for query in test_queries:
        try:
            result = rag_system.query(query)
            results.append({
                "query": query,
                "response": result["response"],
                "language": result["language"],
                "chunks_found": result["metadata"]["chunks_found"]
            })
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e)
            })
    
    return {"test_results": results}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting Multilingual RAG API...")
    
    # Check if knowledge base is set up
    try:
        stats = rag_system.get_system_stats()
        chunk_count = stats.get("vector_database", {}).get("total_chunks", 0)
        
        if chunk_count == 0:
            logger.warning("No documents in knowledge base. Upload PDFs and run /setup to initialize.")
        else:
            logger.info(f"Knowledge base ready with {chunk_count} chunks")
            
    except Exception as e:
        logger.error(f"Startup check failed: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Multilingual RAG API...")

def run_server():
    """Run the FastAPI server"""
    uvicorn.run(
        "api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
