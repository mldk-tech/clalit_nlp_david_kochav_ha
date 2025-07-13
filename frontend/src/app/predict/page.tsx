"use client";
import { useState } from "react";
import { API_ENDPOINTS } from '../../config/api';

export default function PredictPage() {
  const [summary, setSummary] = useState("");
  const [doctorId, setDoctorId] = useState("");
  const [modelVersion, setModelVersion] = useState("latest");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(API_ENDPOINTS.PREDICT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          doctor_id: doctorId,
          summary,
          model_version: modelVersion,
        }),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto py-10 px-4">
      <h1 className="text-2xl font-bold mb-6">Patient Outcome Prediction</h1>
      <form onSubmit={handleSubmit} className="space-y-4 bg-white dark:bg-gray-900 p-6 rounded shadow">
        <div>
          <label className="block font-medium mb-1">Doctor ID</label>
          <input
            type="text"
            className="w-full border rounded px-3 py-2"
            value={doctorId}
            onChange={e => setDoctorId(e.target.value)}
            placeholder="e.g. dr_01"
            required
          />
        </div>
        <div>
          <label className="block font-medium mb-1">Appointment Summary</label>
          <textarea
            className="w-full border rounded px-3 py-2 min-h-[100px]"
            value={summary}
            onChange={e => setSummary(e.target.value)}
            placeholder="Paste or type the clinical summary here..."
            required
          />
        </div>
        <div>
          <label className="block font-medium mb-1">Model Version</label>
          <select
            className="w-full border rounded px-3 py-2"
            value={modelVersion}
            onChange={e => setModelVersion(e.target.value)}
          >
            <option value="latest">Latest</option>
            <option value="xgboost_v1">XGBoost v1</option>
            <option value="random_forest_v1">Random Forest v1</option>
          </select>
        </div>
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          disabled={loading}
        >
          {loading ? "Predicting..." : "Predict Outcome"}
        </button>
      </form>
      {error && <div className="mt-4 text-red-600">{error}</div>}
      {result && (
        <div className="mt-6 bg-gray-100 dark:bg-gray-800 p-4 rounded shadow">
          <h2 className="font-semibold mb-2">Prediction Result</h2>
          <div><b>Prediction:</b> {result.prediction}</div>
          <div><b>Confidence:</b> {result.confidence}</div>
          <div><b>Model Version:</b> {result.model_version}</div>
          <div className="mt-2">
            <b>Features Used:</b>
            <pre className="bg-gray-200 dark:bg-gray-700 p-2 rounded text-xs mt-1">
              {JSON.stringify(result.features_used, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
} 