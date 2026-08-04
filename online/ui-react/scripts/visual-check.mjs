/* Visual regression harness — chụp UI ở nhiều bề rộng VÀ tự kiểm các tiêu chí
   không thể "nhìn cho qua" được:

     1. body/#root KHÔNG được có scrollbar dọc hay ngang
     2. không phần tử nào tràn ra ngoài chiều ngang viewport
     3. nhãn trong WeightPanel không được wrap (đo scrollHeight vs lineHeight)
     4. không có cặp phần tử nào chồng lấn nhau trong hàng WeightRow
     5. cột giữa/phải không được rỗng trơn (phải có empty-state hoặc nội dung)

   Chạy: node scripts/visual-check.mjs [--url http://127.0.0.1:5173]        */

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const url = process.argv.includes("--url") ? process.argv[process.argv.indexOf("--url") + 1] : "http://127.0.0.1:5173";
const outDir = resolve(process.cwd(), "screenshots");
mkdirSync(outDir, { recursive: true });

const VIEWPORTS = [
  { name: "1920", width: 1920, height: 1080 },
  { name: "1708", width: 1708, height: 1000 },
  { name: "1440", width: 1440, height: 900 },
  { name: "1280", width: 1280, height: 800 },
  { name: "mobile", width: 390, height: 844 },
];

/** Chạy trong trang: trả về mọi vi phạm tìm được. */
function auditPage() {
  const problems = [];
  const doc = document.documentElement;

  if (doc.scrollHeight > doc.clientHeight + 1) {
    problems.push(`body scroll dọc: scrollHeight=${doc.scrollHeight} > clientHeight=${doc.clientHeight}`);
  }
  if (doc.scrollWidth > doc.clientWidth + 1) {
    problems.push(`body scroll ngang: scrollWidth=${doc.scrollWidth} > clientWidth=${doc.clientWidth}`);
  }

  // Phần tử tràn ngang khỏi viewport.
  const vw = doc.clientWidth;
  for (const el of document.querySelectorAll("body *")) {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;
    if (rect.right > vw + 1.5) {
      const parent = el.closest("[class]");
      // Vùng cuộn ngang có chủ đích (.scroll-x) được phép có con rộng hơn.
      if (el.closest(".scroll-x")) continue;
      problems.push(`tràn ngang: <${el.tagName.toLowerCase()} class="${el.className}"> right=${rect.right.toFixed(0)} > ${vw}${parent ? "" : ""}`);
      if (problems.length > 25) return problems;
    }
  }

  // Nhãn WeightRow bị wrap?
  for (const label of document.querySelectorAll(".weight-row-label, .field-label, .stat-card-label, .nav-tab, .segment")) {
    const style = getComputedStyle(label);
    const lineHeight = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.5;
    if (label.scrollHeight > lineHeight * 1.6) {
      problems.push(`label wrap: "${label.textContent.trim().slice(0, 30)}" scrollH=${label.scrollHeight} lh=${lineHeight.toFixed(1)}`);
    }
  }

  // Chồng lấn trong WeightRow: slider đè lên ô số?
  for (const row of document.querySelectorAll(".weight-row")) {
    const slider = row.querySelector(".slider");
    const numeric = row.querySelector(".numeric-input");
    if (!slider || !numeric) continue;
    const a = slider.getBoundingClientRect();
    const b = numeric.getBoundingClientRect();
    if (a.right > b.left + 0.5) {
      problems.push(`slider đè ô số trong .weight-row: slider.right=${a.right.toFixed(0)} > numeric.left=${b.left.toFixed(0)}`);
    }
  }

  // Cột rỗng thô: panel hiện hữu nhưng không có nội dung lẫn empty-state.
  for (const selector of [".results-panel", ".preview-panel", ".weight-panel"]) {
    const panel = document.querySelector(selector);
    if (!panel) continue;
    if (getComputedStyle(panel).display === "none") continue;
    const body = panel.querySelector(".panel-body, .results-scroll, .empty-state");
    if (!body) {
      problems.push(`${selector} không có panel-body/empty-state — vùng rỗng thô`);
      continue;
    }
    if (body.textContent.trim().length === 0 && !body.querySelector("img, video, svg, input")) {
      problems.push(`${selector} rỗng hoàn toàn`);
    }
  }

  return problems;
}

const browser = await chromium.launch();
let failed = false;

for (const vp of VIEWPORTS) {
  const context = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForTimeout(900);

  await page.screenshot({ path: `${outDir}/${vp.name}-empty.png` });
  const emptyProblems = await page.evaluate(auditPage);

  // Trạng thái có dữ liệu: chạy một search thật.
  await page.fill(".query-textarea", "cảnh báo sạt lở nguy hiểm ven sông");
  await page.click(".btn-primary");
  await page.waitForTimeout(3500);
  await page.screenshot({ path: `${outDir}/${vp.name}-results.png` });
  const resultProblems = await page.evaluate(auditPage);

  // Sau khi có kết quả, rail phải PHẢI có nội dung thật (đã tự chọn kết quả
  // đầu) — không được để một cột rỗng bắt người dùng đoán phải bấm gì.
  const previewVisible = await page.evaluate(() => {
    const panel = document.querySelector(".preview-panel");
    if (!panel || getComputedStyle(panel).display === "none") return "hidden";
    return panel.querySelector(".detail-list, .preview-media") ? "filled" : "empty";
  });
  if (previewVisible === "empty") resultProblems.push("rail phải rỗng sau khi có kết quả");

  const all = [...emptyProblems.map((p) => `[empty] ${p}`), ...resultProblems.map((p) => `[results] ${p}`)];
  if (all.length > 0) {
    failed = true;
    console.log(`\n### ${vp.name} (${vp.width}x${vp.height}) — ${all.length} vấn đề`);
    for (const problem of all.slice(0, 14)) console.log(`   - ${problem}`);
  } else {
    console.log(`\n### ${vp.name} (${vp.width}x${vp.height}) — OK`);
  }

  await context.close();
}

await browser.close();
console.log(failed ? "\nKẾT QUẢ: CÒN VẤN ĐỀ" : "\nKẾT QUẢ: ĐẠT");
process.exit(failed ? 1 : 0);
