"use client";
import { useEffect, useState } from "react";
import { API_ENDPOINTS } from '../../config/api';

interface ClusterSizes {
  [key: string]: number;
}

interface ClusteringMetrics {
  total_clusters: number;
  silhouette_score: number;
  total_records: number;
  cluster_sizes: ClusterSizes;
}

export default function ClustersPage() {
  const [metrics, setMetrics] = useState<ClusteringMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(API_ENDPOINTS.DISEASE_METRICS)
      .then(res => res.json())
      .then(data => setMetrics(data))
      .catch(() => setError("Failed to load clustering metrics"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-xl mx-auto py-10 px-4">
      <h1 className="text-2xl font-bold mb-6">Disease Clustering Metrics</h1>
      {loading && <div>Loading clustering metrics...</div>}
      {error && <div className="text-red-600">{error}</div>}
      {metrics && (
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-900 p-4 rounded shadow">
            <div><b>Total Clusters:</b> {metrics.total_clusters}</div>
            <div><b>Silhouette Score:</b> {metrics.silhouette_score}</div>
            <div><b>Total Records:</b> {metrics.total_records}</div>
          </div>
          <div>
            <h2 className="font-semibold mb-2 mt-4">Cluster Sizes</h2>
            <table className="w-full border bg-white dark:bg-gray-900 rounded shadow">
              <thead>
                <tr className="bg-blue-100 dark:bg-blue-900">
                  <th className="p-2">Cluster Type</th>
                  <th className="p-2">Number of Clusters</th>
                </tr>
              </thead>
              <tbody>
                {metrics.cluster_sizes && Object.entries(metrics.cluster_sizes).map(([type, size]) => (
                  <tr key={type} className="border-t">
                    <td className="p-2 font-semibold">{type}</td>
                    <td className="p-2">{size}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
} 