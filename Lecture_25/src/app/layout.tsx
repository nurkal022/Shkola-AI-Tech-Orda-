import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TechOrda AI Chat",
  description: "Лекция 25: React + Next.js фронтенд с историей и стримингом",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
