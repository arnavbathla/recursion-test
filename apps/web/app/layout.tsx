import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Recursive ARC Engine",
  description: "TRM/HRM-inspired latent-recursive ARC solver",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-[var(--border)]">
          <nav className="max-w-6xl mx-auto flex items-center gap-6 px-6 py-4 text-sm">
            <Link href="/" className="font-bold text-base">
              Recursive ARC Engine
            </Link>
            <Link href="/tasks" className="text-[var(--muted)] hover:text-white">
              Tasks
            </Link>
            <Link href="/evals" className="text-[var(--muted)] hover:text-white">
              Evals
            </Link>
            <Link href="/models" className="text-[var(--muted)] hover:text-white">
              Models
            </Link>
          </nav>
        </header>
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
