"use client";
import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import toast from 'react-hot-toast';
import { API_ENDPOINTS } from '../../config/api';

interface Doctor {
  id: string;
  rank: number;
  cases: number;
  avg_score: number;
  weighted_score: number;
  outlier: boolean;
  outlier_type?: string;
}

interface AnalyticsStatus {
  total_doctors: number;
  outlier_doctors: number;
  good_outliers: number;
  bad_outliers: number;
  average_weighted_score: number;
  data_file_exists: boolean;
}

export default function DoctorsPage() {
  const [rankings, setRankings] = useState<Doctor[]>([]);
  const [outliers, setOutliers] = useState<{good_outliers: Doctor[], bad_outliers: Doctor[]}>({good_outliers: [], bad_outliers: []});
  const [status, setStatus] = useState<AnalyticsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [runningAnalytics, setRunningAnalytics] = useState(false);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [rankingsRes, outliersRes, statusRes] = await Promise.all([
        fetch(API_ENDPOINTS.DOCTOR_RANKINGS),
        fetch(API_ENDPOINTS.DOCTOR_OUTLIERS),
        fetch(API_ENDPOINTS.DOCTOR_ANALYTICS_STATUS)
      ]);

      if (rankingsRes.ok) {
        const rankingsData = await rankingsRes.json();
        setRankings(rankingsData);
      }

      if (outliersRes.ok) {
        const outliersData = await outliersRes.json();
        setOutliers(outliersData);
      }

      if (statusRes.ok) {
        const statusData = await statusRes.json();
        setStatus(statusData);
      }
    } catch (err) {
      setError("Failed to load doctor analytics");
      toast.error("Failed to load doctor analytics");
    } finally {
      setLoading(false);
    }
  };

  const runAnalytics = async () => {
    try {
      setRunningAnalytics(true);
      const response = await fetch(API_ENDPOINTS.DOCTOR_RUN_ANALYTICS, {
        method: "POST"
      });
      
      if (response.ok) {
        toast.success("Analytics pipeline started successfully");
        // Wait a bit then refresh data
        setTimeout(() => {
          fetchData();
        }, 2000);
      } else {
        toast.error("Failed to start analytics pipeline");
      }
    } catch (err) {
      toast.error("Failed to start analytics pipeline");
    } finally {
      setRunningAnalytics(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Prepare chart data for top 10 doctors
  const chartData = rankings.slice(0, 10).map(doctor => ({
    name: doctor.id,
    weighted_score: doctor.weighted_score,
    avg_score: doctor.avg_score,
    cases: doctor.cases
  }));

  return (
    <div className="max-w-7xl mx-auto py-10 px-4">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Doctor Analytics Dashboard</h1>
        <button
          onClick={runAnalytics}
          disabled={runningAnalytics}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg transition-colors"
        >
          {runningAnalytics ? "Running..." : "Run Analytics"}
        </button>
      </div>

      {loading && <div className="text-center py-8">Loading doctor analytics...</div>}
      {error && <div className="text-red-600 bg-red-100 p-4 rounded-lg mb-6">{error}</div>}

      {status && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300">Total Doctors</h3>
            <p className="text-3xl font-bold text-blue-600">{status.total_doctors}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300">Outlier Doctors</h3>
            <p className="text-3xl font-bold text-orange-600">{status.outlier_doctors}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300">Good Outliers</h3>
            <p className="text-3xl font-bold text-green-600">{status.good_outliers}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300">Bad Outliers</h3>
            <p className="text-3xl font-bold text-red-600">{status.bad_outliers}</p>
          </div>
        </div>
      )}

      {/* Chart Section */}
      {chartData.length > 0 && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow mb-8">
          <h2 className="text-xl font-semibold mb-4">Top 10 Doctors - Performance Overview</h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="weighted_score" fill="#3B82F6" name="Weighted Score" />
              <Bar dataKey="avg_score" fill="#10B981" name="Average Score" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Rankings Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow mb-8">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold">Doctor Rankings</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Rank</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Doctor ID</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Cases</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Avg Score</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Weighted Score</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Outlier</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {rankings.map((doctor) => (
                <tr key={doctor.id} className={doctor.outlier ? "bg-yellow-50 dark:bg-yellow-900/20" : ""}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                    {doctor.rank}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 font-semibold">
                    {doctor.id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {doctor.cases}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {doctor.avg_score.toFixed(3)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {doctor.weighted_score.toFixed(3)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {doctor.outlier ? (
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        doctor.outlier_type === 'good' 
                          ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                          : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                      }`}>
                        {doctor.outlier_type}
                      </span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Outliers Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Good Outliers */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-green-600">Exceptional Performers</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-green-50 dark:bg-green-900/20">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-green-700 dark:text-green-300 uppercase tracking-wider">Rank</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-green-700 dark:text-green-300 uppercase tracking-wider">Doctor ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-green-700 dark:text-green-300 uppercase tracking-wider">Weighted Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {outliers.good_outliers.map((doctor) => (
                  <tr key={doctor.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                      {doctor.rank}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {doctor.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {doctor.weighted_score.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Bad Outliers */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-red-600">Underperforming Doctors</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-red-50 dark:bg-red-900/20">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-red-700 dark:text-red-300 uppercase tracking-wider">Rank</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-red-700 dark:text-red-300 uppercase tracking-wider">Doctor ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-red-700 dark:text-red-300 uppercase tracking-wider">Weighted Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {outliers.bad_outliers.map((doctor) => (
                  <tr key={doctor.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                      {doctor.rank}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {doctor.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {doctor.weighted_score.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
} 