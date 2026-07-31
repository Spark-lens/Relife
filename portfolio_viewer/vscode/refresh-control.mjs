import { createElement, Fragment } from "react";

export function RefreshControl({ refreshing, error, onRefresh }) {
  return createElement(
    Fragment,
    null,
    createElement(
      "button",
      {
        type: "button",
        className: "refresh-button",
        disabled: refreshing,
        "aria-busy": refreshing,
        onClick: onRefresh,
      },
      refreshing ? "正在更新…" : "立即更新",
    ),
    error &&
      createElement(
        "span",
        { className: "refresh-error", role: "alert", title: error },
        `更新失败：${error}`,
      ),
  );
}
