import "./globals.css";
import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Medical Data Analytics System",
  description: "Predict patient outcomes, analyze doctors, and explore disease clusters.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 dark:bg-gray-950 min-h-screen">
        <nav className="w-full bg-blue-700 text-white py-3 px-6 flex gap-6 items-center shadow">
          <Link href="/" className="font-bold text-lg tracking-wide">Clalit NLP</Link>
          <Link href="/predict" className="hover:underline">Predict</Link>
          <Link href="/models" className="hover:underline">Models</Link>
          <Link href="/doctors" className="hover:underline">Doctors</Link>
          <Link href="/clusters" className="hover:underline">Clusters</Link>
        </nav>
        <main className="pt-6">{children}</main>
      </body>
    </html>
  );
}
