import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Relife Portfolio",
  description: "A 股与美股独立核算的私人投资组合仪表盘",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
