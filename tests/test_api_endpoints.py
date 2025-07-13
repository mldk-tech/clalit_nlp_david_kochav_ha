#!/usr/bin/env python3
"""
Test script for API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_health_endpoint():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the backend is running.")
        return False

def test_doctor_endpoints():
    """Test doctor analytics endpoints"""
    print("\n👨‍⚕️ Testing doctor analytics endpoints...")
    
    # Test analytics status
    print("📊 Testing analytics status...")
    try:
        response = requests.get(f"{BASE_URL}/doctors/analytics/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Analytics status: {status}")
        else:
            print(f"❌ Analytics status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Analytics status error: {e}")
    
    # Test run analytics
    print("\n🔄 Testing run analytics...")
    try:
        response = requests.post(f"{BASE_URL}/doctors/run-analytics")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Run analytics: {result}")
            
            # Wait a bit for background task
            print("⏳ Waiting for analytics to complete...")
            time.sleep(3)
            
            # Test rankings
            print("\n🏆 Testing doctor rankings...")
            response = requests.get(f"{BASE_URL}/doctors/rankings")
            if response.status_code == 200:
                rankings = response.json()
                print(f"✅ Found {len(rankings)} doctor rankings")
                if rankings:
                    print("Top 3 doctors:")
                    for i, doctor in enumerate(rankings[:3]):
                        print(f"  {i+1}. {doctor['id']}: Score {doctor['weighted_score']:.3f}")
            else:
                print(f"❌ Rankings failed: {response.status_code}")
            
            # Test outliers
            print("\n🎯 Testing outliers...")
            response = requests.get(f"{BASE_URL}/doctors/outliers")
            if response.status_code == 200:
                outliers = response.json()
                print(f"✅ Found {len(outliers.get('good_outliers', []))} good outliers")
                print(f"✅ Found {len(outliers.get('bad_outliers', []))} bad outliers")
            else:
                print(f"❌ Outliers failed: {response.status_code}")
                
        else:
            print(f"❌ Run analytics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Run analytics error: {e}")

def main():
    """Main test function"""
    print("🚀 Starting API endpoint tests...")
    
    # Test health first
    if not test_health_endpoint():
        print("\n❌ Health check failed. Please start the backend server first.")
        print("Run: cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Test doctor endpoints
    test_doctor_endpoints()
    
    print("\n🎉 API endpoint tests completed!")

if __name__ == "__main__":
    main() 