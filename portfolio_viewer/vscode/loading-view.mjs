import { createElement, Fragment } from "react";

import { RefreshControl } from "./refresh-control.mjs";

export function LoadingView(props) {
  return createElement(
    Fragment,
    null,
    createElement(
      "div",
      { className: "loading-actions" },
      createElement(RefreshControl, props),
    ),
    createElement(
      "main",
      null,
      createElement(
        "div",
        { className: "empty-state" },
        "正在读取投资组合…",
      ),
    ),
  );
}
