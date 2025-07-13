# Doctor Analytics Pipeline

This document describes the complete doctor analytics pipeline that analyzes doctor performance, ranks them, and detects outliers.

## Overview

The doctor analytics system provides:
- **Doctor Performance Scoring**: Calculates weighted scores based on patient outcomes
- **Doctor Ranking**: Ranks doctors by performance metrics
- **Outlier Detection**: Identifies exceptional and underperforming doctors
- **Database Integration**: Stores results in PostgreSQL for persistence
- **API Endpoints**: RESTful API for accessing analytics data
- **Frontend Dashboard**: Modern UI with charts and interactive features

## Architecture

### Backend Components

1. **Doctor Model** (`backend/app/models/doctor.py`)
   - SQLAlchemy model for storing doctor analytics data
   - Fields: id, rank, cases, avg_score, weighted_score, outlier, outlier_type

2. **Doctor Analytics Service** (`backend/app/services/doctor_analytics.py`)
   - Service layer for database operations
   - Integrates with existing doctor ranking logic
   - Handles analytics pipeline execution

3. **Doctor API** (`backend/app/api/doctor.py`)
   - RESTful endpoints for doctor analytics
   - Background task support for pipeline execution

4. **Doctor Ranking Logic** (`backend/app/services/outlier_doctors/doctor_ranking.py`)
   - Core analytics algorithm
   - Outcome scoring and outlier detection
   - Report generation

### Frontend Components

1. **Doctors Dashboard** (`frontend/src/app/doctors/page.tsx`)
   - Comprehensive analytics dashboard
   - Interactive charts using Recharts
   - Real-time data updates
   - Toast notifications

## API Endpoints

### GET `/api/v1/doctors/rankings`
Returns all doctor rankings ordered by weighted score.

**Response:**
```json
[
  {
    "id": "dr_03",
    "rank": 1,
    "cases": 41,
    "avg_score": 4.244,
    "weighted_score": 15.862,
    "outlier": true,
    "outlier_type": "good"
  }
]
```

### GET `/api/v1/doctors/{doctor_id}`
Returns detailed information for a specific doctor.

### GET `/api/v1/doctors/outliers`
Returns outlier doctors separated by type.

**Response:**
```json
{
  "good_outliers": [
    {
      "id": "dr_03",
      "rank": 1,
      "cases": 41,
      "avg_score": 4.244,
      "weighted_score": 15.862
    }
  ],
  "bad_outliers": [
    {
      "id": "dr_07",
      "rank": 10,
      "cases": 35,
      "avg_score": 1.457,
      "weighted_score": 5.124
    }
  ]
}
```

### POST `/api/v1/doctors/run-analytics`
Triggers the analytics pipeline in the background.

**Response:**
```json
{
  "message": "Doctor analytics pipeline started in background"
}
```

### GET `/api/v1/doctors/analytics/status`
Returns analytics pipeline status and statistics.

**Response:**
```json
{
  "total_doctors": 10,
  "outlier_doctors": 2,
  "good_outliers": 1,
  "bad_outliers": 1,
  "average_weighted_score": 10.5,
  "data_file_exists": true
}
```

## Analytics Algorithm

### Scoring System

The system uses a 5-point scoring scale based on patient outcomes:

- **Excellent (5 points)**: Complete recovery, no concerns
- **Good (4 points)**: Significant improvement, stable condition
- **Neutral (3 points)**: No change, monitoring required
- **Poor (2 points)**: Condition deteriorated
- **Critical (1 point)**: Severe decline, emergency care needed

### Weighted Score Calculation

```
Weighted Score = Average Score × log(Total Cases + 1)
```

This formula:
- Rewards doctors with higher average scores
- Considers case volume (logarithmic scaling prevents domination by high-volume doctors)
- Balances quality vs. quantity

### Outlier Detection

Uses the Interquartile Range (IQR) method:
- Calculates Q1 (25th percentile) and Q3 (75th percentile)
- Defines outliers as values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- High outliers = exceptional performers
- Low outliers = underperforming doctors

## Database Schema

```sql
CREATE TABLE doctors (
    id VARCHAR PRIMARY KEY,
    rank INTEGER,
    cases INTEGER,
    avg_score FLOAT,
    weighted_score FLOAT,
    outlier BOOLEAN,
    outlier_type VARCHAR
);
```

## Usage Examples

### Running Analytics Pipeline

1. **Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/doctors/run-analytics
```

2. **Via Python:**
```python
from backend.app.services.doctor_analytics import DoctorAnalyticsService
from backend.app.db.session import get_db

db = next(get_db())
service = DoctorAnalyticsService(db)
service.run_analytics_pipeline()
```

### Accessing Results

1. **Get all rankings:**
```bash
curl http://localhost:8000/api/v1/doctors/rankings
```

2. **Get outliers:**
```bash
curl http://localhost:8000/api/v1/doctors/outliers
```

3. **Get specific doctor:**
```bash
curl http://localhost:8000/api/v1/doctors/dr_03
```

### Testing

1. **Test analytics logic:**
```bash
python test_doctor_analytics.py
```

2. **Test API endpoints:**
```bash
python test_api_endpoints.py
```

## Frontend Features

### Dashboard Components

1. **Statistics Cards**
   - Total doctors analyzed
   - Number of outliers
   - Good vs. bad outliers
   - Average weighted score

2. **Performance Chart**
   - Bar chart showing top 10 doctors
   - Weighted score vs. average score comparison
   - Interactive tooltips

3. **Rankings Table**
   - Complete doctor rankings
   - Sortable columns
   - Outlier indicators
   - Color-coded performance

4. **Outlier Sections**
   - Separate tables for good and bad outliers
   - Performance metrics
   - Visual indicators

### Interactive Features

- **Run Analytics Button**: Triggers pipeline execution
- **Real-time Updates**: Automatic data refresh
- **Toast Notifications**: Success/error feedback
- **Responsive Design**: Works on all screen sizes

## Setup Instructions

### Prerequisites

1. PostgreSQL database running
2. Python dependencies installed
3. Node.js and npm for frontend

### Database Setup

1. **Run migrations:**
```bash
alembic upgrade head
```

2. **Verify table creation:**
```sql
\d doctors;
```

### Backend Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Start server:**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Start development server:**
```bash
npm run dev
```

## Data Requirements

The system expects a JSON file `doctor_appointment_summaries.json` with the following structure:

```json
[
  {
    "doctor_id": "dr_01",
    "future_outcome": "Patient is feeling much better and has resumed daily activities.",
    "appointment_date": "2024-01-15",
    "patient_id": "patient_001"
  }
]
```

## Performance Considerations

- **Background Processing**: Analytics pipeline runs asynchronously
- **Database Indexing**: Consider adding indexes on frequently queried columns
- **Caching**: Implement Redis caching for frequently accessed data
- **Batch Processing**: For large datasets, consider batch processing

## Monitoring and Logging

- **Application Logs**: Check backend logs for pipeline execution
- **Database Monitoring**: Monitor query performance
- **API Metrics**: Track endpoint usage and response times
- **Error Handling**: Comprehensive error logging and user feedback

## Future Enhancements

1. **Advanced Analytics**
   - Time-series analysis
   - Trend detection
   - Predictive modeling

2. **Enhanced UI**
   - More chart types
   - Filtering and search
   - Export functionality

3. **Performance Optimizations**
   - Database query optimization
   - Caching strategies
   - Parallel processing

4. **Additional Metrics**
   - Patient satisfaction scores
   - Treatment effectiveness
   - Cost analysis

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify PostgreSQL is running
   - Check connection settings in `settings.py`
   - Ensure database exists

2. **Data File Not Found**
   - Verify `doctor_appointment_summaries.json` exists in root directory
   - Check file permissions

3. **Migration Errors**
   - Run `alembic current` to check migration status
   - Use `alembic downgrade` to rollback if needed

4. **API Errors**
   - Check server logs for detailed error messages
   - Verify endpoint URLs and request formats

### Debug Mode

Enable debug mode in `backend/app/config/settings.py`:
```python
debug = True
```

This provides detailed error messages and stack traces. 