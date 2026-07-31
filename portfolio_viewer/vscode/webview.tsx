import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import "../app/globals.css";
import { PortfolioDashboard } from "../app/components/PortfolioDashboard";
import type { DashboardPayload } from "../app/portfolio-types";

declare function acquireVsCodeApi(): {
  postMessage(message: unknown): void;
};

const vscode = acquireVsCodeApi();

function App() {
  const [data, setData] = useState<DashboardPayload | null>(null);

  useEffect(() => {
    const receive = (event: MessageEvent) => {
      if (event.data?.type === "portfolio") setData(event.data.data);
    };
    window.addEventListener("message", receive);
    vscode.postMessage({ type: "ready" });
    return () => window.removeEventListener("message", receive);
  }, []);

  return data ? (
    <PortfolioDashboard data={data} />
  ) : (
    <main>
      <div className="empty-state">正在读取投资组合…</div>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
