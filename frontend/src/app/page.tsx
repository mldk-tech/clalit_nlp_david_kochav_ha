import Link from "next/link";

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6 sm:p-10">
      <header className="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2 sm:mb-0">Clalit NLP Dashboard</h1>
        <nav className="flex gap-4">
          <Link href="/clusters" className="text-blue-600 hover:underline">Clusters</Link>
          <Link href="/doctors" className="text-blue-600 hover:underline">Doctors</Link>
          <Link href="/models" className="text-blue-600 hover:underline">Models</Link>
          <Link href="/predict" className="text-blue-600 hover:underline">Predict</Link>
        </nav>
      </header>
      <main className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* System Health Card */}
        <section className="col-span-1 bg-white dark:bg-gray-800 rounded-lg shadow p-6 flex flex-col gap-2">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-2">System Health</h2>
          <div className="flex flex-col gap-1 text-gray-700 dark:text-gray-300 text-sm">
            <div>Status: <span className="font-bold text-green-600">Healthy</span></div>
            <div>Last Checked: <span>Just now</span></div>
            <div>Active Models: <span className="font-mono">2</span></div>
            <div>Database: <span className="font-mono">Connected</span></div>
          </div>
          <Link href="/api/health" className="mt-2 text-blue-500 hover:underline text-xs">View Details</Link>
        </section>
        {/* Models & Metrics Card */}
        <section className="col-span-1 bg-white dark:bg-gray-800 rounded-lg shadow p-6 flex flex-col gap-2">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-2">Models & Metrics</h2>
          <div className="flex flex-col gap-1 text-gray-700 dark:text-gray-300 text-sm">
            <div>Best Model: <span className="font-mono">Random Forest v1</span></div>
            <div>Accuracy: <span className="font-mono">0.85</span></div>
            <div>Last Trained: <span>2024-01-01</span></div>
            <div>Predictions: <span className="font-mono">1,250</span></div>
          </div>
          <Link href="/models" className="mt-2 text-blue-500 hover:underline text-xs">View All Models</Link>
        </section>
        {/* Quick Prediction Card */}
        <section className="col-span-1 bg-white dark:bg-gray-800 rounded-lg shadow p-6 flex flex-col gap-2">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-2">Quick Prediction</h2>
          <form className="flex flex-col gap-2">
            <textarea
              className="rounded border border-gray-300 dark:border-gray-700 p-2 text-sm bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-100"
              rows={3}
              placeholder="Paste appointment summary here..."
              disabled
            />
            <button
              type="button"
              className="bg-blue-600 text-white rounded px-4 py-2 font-semibold hover:bg-blue-700 disabled:opacity-50"
              disabled
            >
              Predict (API integration coming soon)
            </button>
          </form>
        </section>
      </main>
      <footer className="mt-12 text-center text-xs text-gray-500 dark:text-gray-400">
        &copy; {new Date().getFullYear()} Clalit NLP Dashboard. Powered by Next.js &amp; FastAPI.
      </footer>
    </div>
  );
}
