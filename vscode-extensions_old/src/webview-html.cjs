function webviewHtml({ kind, script, style, cspSource, nonce }) {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${cspSource} data:; style-src ${cspSource}; script-src 'nonce-${nonce}';">
  <link rel="stylesheet" href="${style}">
  <title>Relife</title>
</head>
<body data-view="${kind}"><div id="root"></div><script type="module" nonce="${nonce}" src="${script}"></script></body>
</html>`;
}

module.exports = { webviewHtml };
