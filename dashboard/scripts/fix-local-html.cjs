const fs = require("node:fs");
const path = require("node:path");

const indexPath = path.resolve(__dirname, "../../artifacts/dashboard/index.html");
let html = fs.readFileSync(indexPath, "utf8");

html = html.replace(
  /<script type="module" crossorigin src="(\.\/assets\/[^"]+\.js)"><\/script>/,
  '<script defer src="$1"></script>',
);
html = html.replace(/\s+crossorigin(?= href="\.\/assets\/[^"]+\.css")/g, "");

fs.writeFileSync(indexPath, html);
