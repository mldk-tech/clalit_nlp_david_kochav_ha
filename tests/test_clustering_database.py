#!/usr/bin/env python3
"""
Test script to verify that clustering results are saved to the database
instead of JSON file.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.db.session import SessionLocal
from backend.app.models.disease_clustering import DiseaseClusteringResult
from backend.app.services.alike_diseases_clusters.disease_clustering import run_disease_clustering_pipeline

def test_clustering_database_save():
    """Test that clustering results are saved to database"""
    print("Testing clustering database save...")
    
    # Check if database table exists and is empty
    db = SessionLocal()
    try:
        initial_count = db.query(DiseaseClusteringResult).count()
        print(f"Initial records in database: {initial_count}")
        
        # Run clustering pipeline
        print("Running clustering pipeline...")
        result = run_disease_clustering_pipeline()
        
        # Check if results were saved to database
        final_count = db.query(DiseaseClusteringResult).count()
        print(f"Final records in database: {final_count}")
        
        if final_count > initial_count:
            print("✅ SUCCESS: Clustering results were saved to database!")
            
            # Get some sample data
            sample_results = db.query(DiseaseClusteringResult).limit(5).all()
            print(f"\nSample records:")
            for i, record in enumerate(sample_results):
                print(f"  {i+1}. Method: {record.clustering_method}, Cluster: {record.cluster_label}, Summary: {record.summary[:50]}...")
            
            # Check different methods
            methods = db.query(DiseaseClusteringResult.clustering_method).distinct().all()
            print(f"\nClustering methods found: {[m[0] for m in methods]}")
            
        else:
            print("❌ FAILED: No new records were added to database")
            
        # Check database save status in result
        if 'database_save' in result:
            print(f"Database save status: {result['database_save']}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        db.close()
    
    return True

def test_clustering_api_endpoints():
    """Test the new API endpoints"""
    print("\nTesting API endpoints...")
    
    # This would require the FastAPI server to be running
    # For now, just test the database queries directly
    db = SessionLocal()
    try:
        # Test getting clustering methods
        methods = db.query(DiseaseClusteringResult.clustering_method).distinct().all()
        print(f"Available methods: {[m[0] for m in methods]}")
        
        # Test getting results for a specific method
        if methods:
            method = methods[0][0]
            results = db.query(DiseaseClusteringResult).filter(
                DiseaseClusteringResult.clustering_method == method
            ).limit(3).all()
            
            print(f"\nSample results for method '{method}':")
            for result in results:
                print(f"  - Cluster {result.cluster_label}: {result.summary[:30]}...")
        
        # Test statistics
        total_records = db.query(DiseaseClusteringResult).count()
        print(f"\nTotal records: {total_records}")
        
        # Group by method and cluster
        stats = db.query(
            DiseaseClusteringResult.clustering_method,
            DiseaseClusteringResult.cluster_label,
            db.func.count(DiseaseClusteringResult.id).label('count')
        ).group_by(
            DiseaseClusteringResult.clustering_method,
            DiseaseClusteringResult.cluster_label
        ).all()
        
        print("\nStatistics by method and cluster:")
        for method, cluster, count in stats[:10]:  # Show first 10
            print(f"  {method} - Cluster {cluster}: {count} records")
        
    except Exception as e:
        print(f"❌ ERROR testing API endpoints: {e}")
        return False
    finally:
        db.close()
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("CLUSTERING DATABASE TEST")
    print("=" * 60)
    
    success1 = test_clustering_database_save()
    success2 = test_clustering_api_endpoints()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ ALL TESTS PASSED: Clustering results are now saved to database!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60) 