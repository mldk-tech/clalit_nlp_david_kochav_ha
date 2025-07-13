import os
import json
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.services.alike_diseases_clusters.disease_clustering import DiseaseClusteringEngine, run_disease_clustering_pipeline
from backend.app.models.disease_clustering import DiseaseClusteringResult

router = APIRouter()

@router.post("/run-disease-clustering", summary="Run the disease clustering script")
def run_disease_clustering():
    """
    Run the disease clustering pipeline and save results to database.
    """
    try:
        result = run_disease_clustering_pipeline()
        return {"status": "success", "message": "Disease clustering completed and saved to database.", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running clustering: {str(e)}")

@router.get("/clustering-results", summary="Get clustering results from database")
def get_clustering_results(
    method: Optional[str] = Query(None, description="Filter by clustering method"),
    db: Session = Depends(get_db)
):
    """
    Retrieve clustering results from the database.
    """
    try:
        query = db.query(DiseaseClusteringResult)
        if method:
            query = query.filter(DiseaseClusteringResult.clustering_method == method)
        
        results = query.all()
        
        # Group by method and cluster
        grouped_results = {}
        for result in results:
            if result.clustering_method not in grouped_results:
                grouped_results[result.clustering_method] = {}
            
            if result.cluster_label not in grouped_results[result.clustering_method]:
                grouped_results[result.clustering_method][result.cluster_label] = []
            
            grouped_results[result.clustering_method][result.cluster_label].append({
                'id': str(result.id),
                'summary': result.summary,
                'diseases': result.diseases,
                'symptoms': result.symptoms,
                'treatments': result.treatments,
                'outcomes': result.outcomes,
                'severity_score': result.severity_score,
                'complexity_score': result.complexity_score,
                'doctor_id': result.doctor_id,
                'algorithm': result.algorithm,
                'parameters': result.parameters,
                'created_at': result.created_at.isoformat() if result.created_at else None
            })
        
        return {
            "status": "success",
            "total_records": len(results),
            "methods": list(grouped_results.keys()),
            "results": grouped_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving clustering results: {str(e)}")

@router.get("/clustering-methods", summary="Get available clustering methods")
def get_clustering_methods(db: Session = Depends(get_db)):
    """
    Get list of available clustering methods from database.
    """
    try:
        methods = db.query(DiseaseClusteringResult.clustering_method).distinct().all()
        return {
            "status": "success",
            "methods": [method[0] for method in methods]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving clustering methods: {str(e)}")

@router.get("/clustering-stats", summary="Get clustering statistics")
def get_clustering_stats(db: Session = Depends(get_db)):
    """
    Get statistics about clustering results.
    """
    try:
        # Get total records
        total_records = db.query(DiseaseClusteringResult).count()
        
        # Get records per method
        method_stats = db.query(
            DiseaseClusteringResult.clustering_method,
            DiseaseClusteringResult.cluster_label,
            db.func.count(DiseaseClusteringResult.id).label('count')
        ).group_by(
            DiseaseClusteringResult.clustering_method,
            DiseaseClusteringResult.cluster_label
        ).all()
        
        # Group by method
        stats = {}
        for method, cluster, count in method_stats:
            if method not in stats:
                stats[method] = {}
            stats[method][cluster] = count
        
        return {
            "status": "success",
            "total_records": total_records,
            "method_statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving clustering statistics: {str(e)}") 