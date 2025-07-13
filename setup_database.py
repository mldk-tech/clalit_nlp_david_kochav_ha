#!/usr/bin/env python3
"""
Database setup script for Clalit NLP Doctor Analytics
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database():
    """Setup PostgreSQL database and user"""
    print("🔧 Setting up PostgreSQL database...")
    
    # Database configuration
    DB_NAME = "clalit"
    DB_USER = "postgres"
    DB_PASSWORD = "postgres"  # Change this to your actual password
    DB_HOST = "localhost"
    DB_PORT = "5433"
    
    try:
        # Connect to PostgreSQL server (without specifying database)
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database="postgres"  # Connect to default postgres database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"📊 Creating database '{DB_NAME}'...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"✅ Database '{DB_NAME}' created successfully")
        else:
            print(f"✅ Database '{DB_NAME}' already exists")
        
        cursor.close()
        conn.close()
        
        # Test connection to the new database
        test_conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        test_conn.close()
        print("✅ Database connection test successful")
        
        # Set environment variable
        os.environ['DATABASE_URL'] = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        print(f"✅ Environment variable DATABASE_URL set")
        
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Database connection failed: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure PostgreSQL is installed and running")
        print("2. Check if PostgreSQL service is started:")
        print("   - Windows: services.msc -> PostgreSQL")
        print("   - Linux: sudo systemctl start postgresql")
        print("3. Verify your PostgreSQL password")
        print("4. Try connecting manually: psql -U postgres -h localhost")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_migrations():
    """Run Alembic migrations"""
    print("\n🔄 Running database migrations...")
    try:
        import subprocess
        result = subprocess.run(['alembic', 'upgrade', 'head'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Migrations completed successfully")
            return True
        else:
            print(f"❌ Migration failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False

def test_doctor_analytics():
    """Test the doctor analytics functionality with database"""
    print("\n🧪 Testing doctor analytics with database...")
    
    try:
        # Import after setting up environment
        sys.path.append('backend')
        from backend.app.services.doctor_analytics import DoctorAnalyticsService
        from backend.app.db.session import get_db
        
        # Get database session
        db = next(get_db())
        service = DoctorAnalyticsService(db)
        
        # Run analytics pipeline
        print("📊 Running analytics pipeline...")
        service.run_analytics_pipeline()
        
        # Test data retrieval
        print("📋 Testing data retrieval...")
        rankings = service.get_all_rankings()
        print(f"✅ Found {len(rankings)} doctor rankings in database")
        
        outliers = service.get_outlier_doctors()
        print(f"✅ Found {len(outliers.get('good_outliers', []))} good outliers")
        print(f"✅ Found {len(outliers.get('bad_outliers', []))} bad outliers")
        
        status = service.get_analytics_status()
        print(f"✅ Analytics status: {status}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Error testing analytics: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Starting Clalit NLP Doctor Analytics Database Setup")
    print("=" * 60)
    
    # Step 1: Setup database
    if not setup_database():
        print("\n❌ Database setup failed. Please fix the issues above.")
        return False
    
    # Step 2: Run migrations
    if not run_migrations():
        print("\n❌ Migration failed. Please check the errors above.")
        return False
    
    # Step 3: Test analytics
    if not test_doctor_analytics():
        print("\n❌ Analytics test failed. Please check the errors above.")
        return False
    
    print("\n🎉 Database setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the backend server:")
    print("   python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000")
    print("2. Start the frontend server:")
    print("   cd frontend && npm run dev")
    print("3. Visit the doctors dashboard at: http://localhost:3000/doctors")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 