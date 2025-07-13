// API configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// API endpoints
export const API_ENDPOINTS = {
  // Doctor analytics
  DOCTOR_RANKINGS: `${API_BASE_URL}/api/v1/doctors/rankings`,
  DOCTOR_OUTLIERS: `${API_BASE_URL}/api/v1/doctors/outliers`,
  DOCTOR_ANALYTICS_STATUS: `${API_BASE_URL}/api/v1/doctors/analytics/status`,
  DOCTOR_RUN_ANALYTICS: `${API_BASE_URL}/api/v1/doctors/run-analytics`,
  DOCTOR_DETAILS: (id: string) => `${API_BASE_URL}/api/v1/doctors/${id}`,
  
  // Models
  MODELS: `${API_BASE_URL}/api/v1/models`,
  MODELS_TRAIN: `${API_BASE_URL}/api/v1/models/train`,
  
  // Predictions
  PREDICT: `${API_BASE_URL}/api/v1/predict`,
  
  // Metrics
  DISEASE_METRICS: `${API_BASE_URL}/api/v1/metrics/diseases`,
  
  // Health check
  HEALTH: `${API_BASE_URL}/api/v1/health`,
} as const; 