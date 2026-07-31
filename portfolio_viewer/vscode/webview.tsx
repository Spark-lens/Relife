import { useEffect, useReducer } from "react";
import { createRoot } from "react-dom/client";

import "../app/globals.css";
import { PortfolioDashboard } from "../app/components/PortfolioDashboard";
import type { DashboardPayload } from "../app/portfolio-types";
import { LoadingView } from "./loading-view.mjs";
import { RefreshControl } from "./refresh-control.mjs";
import {
  createRefreshRequest,
  initialWebviewState,
  reduceWebviewState,
} from "./webview-state.mjs";

declare function acquireVsCodeApi(): {
  postMessage(message: unknown): void;
};

const vscode = acquireVsCodeApi();

function App() {
  const [state, dispatch] = useReducer(
    reduceWebviewState,
    initialWebviewState,
  );

  useEffect(() => {
    const receive = (event: MessageEvent) => dispatch(event.data);
    window.addEventListener("message", receive);
    vscode.postMessage({ type: "ready" });
    return () => window.removeEventListener("message", receive);
  }, []);

  function refresh() {
    dispatch({ type: "refresh-start" });
    vscode.postMessage(createRefreshRequest());
  }

  return (
    state.data ? (
      <PortfolioDashboard
        data={state.data as DashboardPayload}
        toolbar={
          <RefreshControl
            refreshing={state.refreshing}
            error={state.error}
            onRefresh={refresh}
          />
        }
      />
    ) : (
      <LoadingView
        refreshing={state.refreshing}
        error={state.error}
        onRefresh={refresh}
      />
    )
  );
}

createRoot(document.getElementById("root")!).render(<App />);
