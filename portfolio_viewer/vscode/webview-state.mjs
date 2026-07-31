export const initialWebviewState = {
  data: null,
  refreshing: false,
  error: null,
};

export function createRefreshRequest() {
  return { type: "refresh" };
}

export function reduceWebviewState(state, message) {
  switch (message?.type) {
    case "portfolio":
      return { ...state, data: message.data };
    case "refresh-start":
      return { ...state, refreshing: true, error: null };
    case "refresh-success":
      return { ...state, refreshing: false };
    case "refresh-error":
      return { ...state, refreshing: false, error: message.message };
    default:
      return state;
  }
}
