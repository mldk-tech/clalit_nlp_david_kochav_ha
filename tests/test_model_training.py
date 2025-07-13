#!/usr/bin/env python3
"""
Test script for model training functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
from datetime import datetime
from backend.app.api.model import (
    train_model, 
    get_training_status, 
    list_training_jobs,
    ModelTrainingRequest,
    training_jobs
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_model_training():
    """Test the model training functionality"""
    try:
        logger.info("Testing model training functionality...")
        
        # Test 1: Start a training job
        logger.info("Test 1: Starting training job...")
        
        # Create a mock request
        request = ModelTrainingRequest(
            model_type="both",
            hyperparameters={
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10
                },
                "xgboost": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1
                }
            },
            data_version="v1.0"
        )
        
        # Mock background tasks (in real scenario, this would be FastAPI's BackgroundTasks)
        class MockBackgroundTasks:
            def __init__(self):
                self.tasks = []
            
            def add_task(self, func, *args, **kwargs):
                self.tasks.append((func, args, kwargs))
                logger.info(f"Added background task: {func.__name__}")
        
        background_tasks = MockBackgroundTasks()
        
        # Mock database session
        class MockDBSession:
            pass
        
        db = MockDBSession()
        
        # Test the training endpoint
        try:
            response = await train_model(request, background_tasks, db)
            logger.info(f"Training response: {response}")
            
            # Verify response structure
            assert response.status == "training_started"
            assert response.job_id is not None
            assert response.model_type == "both"
            assert response.estimated_completion is not None
            assert response.message == "Training pipeline started successfully"
            
            job_id = response.job_id
            logger.info(f"✅ Training job started successfully with ID: {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start training job: {e}")
            raise
        
        # Test 2: Check training job status
        logger.info("Test 2: Checking training job status...")
        
        try:
            status_response = await get_training_status(job_id)
            logger.info(f"Status response: {status_response}")
            
            # Verify status structure
            assert status_response["job_id"] == job_id
            assert status_response["status"] in ["started", "running", "completed", "failed"]
            assert status_response["model_type"] == "both"
            assert "progress" in status_response
            assert "message" in status_response
            
            logger.info(f"✅ Training status retrieved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to get training status: {e}")
            raise
        
        # Test 3: List all training jobs
        logger.info("Test 3: Listing all training jobs...")
        
        try:
            jobs_response = await list_training_jobs()
            logger.info(f"Jobs response: {jobs_response}")
            
            # Verify jobs list structure
            assert "jobs" in jobs_response
            assert isinstance(jobs_response["jobs"], list)
            
            # Should have at least one job (the one we just created)
            assert len(jobs_response["jobs"]) >= 1
            
            # Find our job in the list
            our_job = next((job for job in jobs_response["jobs"] if job["job_id"] == job_id), None)
            assert our_job is not None
            assert our_job["model_type"] == "both"
            
            logger.info(f"✅ Training jobs list retrieved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to list training jobs: {e}")
            raise
        
        # Test 4: Test invalid model type
        logger.info("Test 4: Testing invalid model type...")
        
        try:
            invalid_request = ModelTrainingRequest(
                model_type="invalid_model",
                hyperparameters={},
                data_version="v1.0"
            )
            
            # This should raise an HTTPException
            await train_model(invalid_request, background_tasks, db)
            logger.error("❌ Expected HTTPException for invalid model type, but none was raised")
            
        except Exception as e:
            if "Invalid model type" in str(e):
                logger.info(f"✅ Correctly rejected invalid model type: {e}")
            else:
                logger.error(f"❌ Unexpected error for invalid model type: {e}")
                raise
        
        # Test 5: Test non-existent job status
        logger.info("Test 5: Testing non-existent job status...")
        
        try:
            await get_training_status("non-existent-job-id")
            logger.error("❌ Expected HTTPException for non-existent job, but none was raised")
            
        except Exception as e:
            if "Training job not found" in str(e):
                logger.info(f"✅ Correctly rejected non-existent job: {e}")
            else:
                logger.error(f"❌ Unexpected error for non-existent job: {e}")
                raise
        
        logger.info("✅ All model training functionality tests completed successfully!")
        
        # Clean up test data
        if job_id in training_jobs:
            del training_jobs[job_id]
            logger.info(f"Cleaned up test job {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_model_training()) 