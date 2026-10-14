import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "/Users/lihd/discovery/paleomagnetic-r3-workflow/dist/paleomagnetic-r357-calibrated-workflow-full";
const outputDir = path.join(root, "outputs/rockmag_mixed_label_multitarget_20260717");
function xmlSafe(value) {
  if (typeof value === "string") return value.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFFFE\uFFFF]/g, " ");
  if (Array.isArray(value)) return value.map(xmlSafe);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, xmlSafe(item)]));
  return value;
}

const summary = xmlSafe(JSON.parse(await fs.readFile(path.join(outputDir, "evaluation_summary.json"), "utf8")));

async function readJsonl(file) {
  return (await fs.readFile(file, "utf8"))
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => xmlSafe(JSON.parse(line)));
}

const targets = await readJsonl(path.join(outputDir, "target_results.jsonl"));
const papers = await readJsonl(path.join(outputDir, "paper_summary.jsonl"));
const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("结果摘要");
const targetSheet = workbook.worksheets.add("目标明细");
const paperSheet = workbook.worksheets.add("论文汇总");
const mismatchSheet = workbook.worksheets.add("ROCKMAG-R3不一致");
const protocolSheet = workbook.worksheets.add("口径与溯源");

const colors = {
  ink: "#17212B",
  muted: "#52606D",
  teal: "#087E8B",
  tealLight: "#DDF3F2",
  green: "#2F855A",
  greenLight: "#E5F4EA",
  coral: "#C84B31",
  coralLight: "#FCE8E3",
  amber: "#A46400",
  amberLight: "#FFF1CC",
  blueLight: "#E8F0FE",
  gray: "#E5E9ED",
  grayLight: "#F6F8FA",
  white: "#FFFFFF",
};

function title(sheet, range, text) {
  range.merge();
  range.values = [[text]];
  range.format = {
    fill: colors.ink,
    font: { bold: true, color: colors.white, size: 18 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 32;
}

function section(range) {
  range.format = {
    fill: colors.teal,
    font: { bold: true, color: colors.white },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 23;
}

function headers(range) {
  range.format = {
    fill: colors.ink,
    font: { bold: true, color: colors.white },
    wrapText: true,
    verticalAlignment: "center",
  };
  range.format.rowHeight = 34;
}

function baseSheet(sheet) {
  sheet.showGridLines = false;
}

for (const sheet of [summarySheet, targetSheet, paperSheet, mismatchSheet, protocolSheet]) {
  baseSheet(sheet);
}

title(summarySheet, summarySheet.getRange("A1:D1"), "混合标签多目标样本：ROCKMAG-first 盲测");
summarySheet.getRange("A2:D2").merge();
summarySheet.getRange("A2").values = [["同一篇论文内至少两个目标，且历史 R3 同时包含 0 和 1；测试日期 2026-07-17"]];
summarySheet.getRange("A2:D2").format = { font: { color: colors.muted, italic: true }, wrapText: true };

summarySheet.getRange("A4:B4").values = [["测试范围", "值"]];
section(summarySheet.getRange("A4:B4"));
summarySheet.getRange("A5:A13").values = [
  ["目标数"],
  ["论文数"],
  ["正文覆盖率"],
  ["抽取为非空 ROCKMAG"],
  ["明确判定数（PRESENT 或 ABSENT）"],
  ["明确判定覆盖率"],
  ["历史 R3=1"],
  ["历史 ROCKMAG 非空"],
  ["历史 ROCKMAG 非空直接派生 R3 的一致率"],
];
summarySheet.getRange("B5:B13").formulas = [
  ["=COUNTA('目标明细'!$A$2:$A$279)"],
  ["=COUNTA('论文汇总'!$A$2:$A$62)"],
  ["=1"],
  ["=COUNTIF('目标明细'!$Q$2:$Q$279,\"PRESENT\")"],
  ["=SUM('目标明细'!$W$2:$W$279)"],
  ["=B9/B5"],
  ["=COUNTIF('目标明细'!$O$2:$O$279,1)"],
  ["=SUM('目标明细'!$N$2:$N$279)"],
  ["=AVERAGE('目标明细'!$P$2:$P$279)"],
];
summarySheet.getRange("B7:B7").format.numberFormat = "0.00%";
summarySheet.getRange("B10:B10").format.numberFormat = "0.00%";
summarySheet.getRange("B13:B13").format.numberFormat = "0.00%";

summarySheet.getRange("A15:D15").values = [["方法比较", "R3 准确率", "相对原基线", "相对成对排序"]];
section(summarySheet.getRange("A15:D15"));
summarySheet.getRange("A16:A18").values = [
  ["原扩展基线（表格 48.56%）"],
  ["原成对排序（表格 56.12%）"],
  ["ROCKMAG-first 强制二分类（拒判记为 0）"],
];
summarySheet.getRange("B16:D18").formulas = [
  ["=AVERAGE('目标明细'!$AJ$2:$AJ$279)", "=0", "=B16-B17"],
  ["=AVERAGE('目标明细'!$AM$2:$AM$279)", "=B17-B16", "=0"],
  ["=AVERAGE('目标明细'!$V$2:$V$279)", "=B18-B16", "=B18-B17"],
];
summarySheet.getRange("B16:D18").format.numberFormat = "0.00%";
summarySheet.getRange("A18:D18").format.fill = colors.amberLight;

summarySheet.getRange("A20:D20").values = [["评测对象", "全量准确率", "明确判定准确率", "说明"]];
section(summarySheet.getRange("A20:D20"));
summarySheet.getRange("A21:D22").values = [
  [
    "ROCKMAG 是否非空",
    summary.rockmag_presence_prediction.full_set.accuracy,
    summary.rockmag_presence_prediction.decided_only.accuracy,
    "全量口径把所有拒判按空值处理；明确判定口径只含 PRESENT/ABSENT",
  ],
  [
    "由预测 ROCKMAG 直接派生 R3",
    summary.direct_r3_prediction.full_set.accuracy,
    summary.direct_r3_prediction.decided_only.accuracy,
    "历史 ROCKMAG 是否非空与 R3 有 18/278 行不一致",
  ],
];
summarySheet.getRange("B21:C22").format.numberFormat = "0.00%";
summarySheet.getRange("D21:D22").format.wrapText = true;

summarySheet.getRange("A24:D24").merge();
summarySheet.getRange("A24").values = [[
  "结论：强制二分类准确率为 62.95%，比表格中的 48.56% 和 56.12% 分别高 14.39 与 6.83 个百分点；但真正明确得到 ROCKMAG 存在/不存在的覆盖率仅 48.20%，且非空字段的精确字符串一致率很低，因此当前版本尚不能替代原工作流。",
]];
summarySheet.getRange("A24:D24").format = {
  fill: colors.coralLight,
  font: { bold: true, color: colors.coral },
  wrapText: true,
  verticalAlignment: "center",
};
summarySheet.getRange("A24:D24").format.rowHeight = 58;
summarySheet.getRange("A4:B13").format.borders = { preset: "inside", style: "thin", color: colors.gray };
summarySheet.getRange("A15:D18").format.borders = { preset: "inside", style: "thin", color: colors.gray };
summarySheet.getRange("A20:D22").format.borders = { preset: "inside", style: "thin", color: colors.gray };
summarySheet.getRange("A1:D24").format.font.name = "Arial";
summarySheet.getRange("A1:A24").format.columnWidth = 43;
summarySheet.getRange("B1:C24").format.columnWidth = 18;
summarySheet.getRange("D1:D24").format.columnWidth = 58;
summarySheet.freezePanes.freezeRows(2);

const targetHeaders = [
  "REFNO", "RESULTNO", "ROCKUNITNO", "ROCKNAME", "RLAT", "RLONG", "COMPONENT", "DEC", "INC", "PLAT", "PLONG", "PTYPE",
  "历史 ROCKMAG", "历史 ROCKMAG 非空", "历史 R3", "ROCKMAG存在=R3", "预测状态", "预测 ROCKMAG", "候选 ROCKMAG", "直接派生 R3",
  "ROCKMAG存在判断正确", "R3正确", "明确判定", "置信度", "定位结果", "元数据唯一", "Component唯一", "同篇同名重复", "方法", "载磁矿物", "磁性域", "证据页", "证据摘录",
  "原基线概率", "原基线预测", "原基线正确", "成对排序概率", "成对排序预测", "成对排序正确",
];
targetSheet.getRange("A1:AM1").values = [targetHeaders];
headers(targetSheet.getRange("A1:AM1"));
const targetValues = targets.map((row) => [
  row.REFNO, row.RESULTNO, row.ROCKUNITNO, row.ROCKNAME, row.RLAT, row.RLONG, row.COMPONENT, row.DEC, row.INC, row.PLAT, row.PLONG, row.PTYPE,
  row.historical_rockmag, null, row.historical_r3, null, row.prediction_status, row.predicted_rockmag, row.candidate_rockmag, null, null, null, null,
  row.confidence, row.locator_resolution, row.locator_metadata_unique, row.locator_component_unique, row.duplicate_same_name_within_paper,
  row.methods, row.carriers, row.domains, row.evidence_pages, row.evidence_excerpt,
  row.old_baseline_probability, null, null, row.old_pairwise_probability, null, null,
]);
targetSheet.getRange(`A2:AM${targets.length + 1}`).values = targetValues;
for (let row = 2; row <= targets.length + 1; row += 1) {
  targetSheet.getRange(`N${row}`).formulas = [[`=--(LEN(M${row})>0)`]];
  targetSheet.getRange(`P${row}`).formulas = [[`=--(N${row}=O${row})`]];
  targetSheet.getRange(`T${row}:W${row}`).formulas = [[
    `=--(Q${row}=\"PRESENT\")`, `=--(T${row}=N${row})`, `=--(T${row}=O${row})`, `=--(OR(Q${row}=\"PRESENT\",Q${row}=\"ABSENT\"))`,
  ]];
  targetSheet.getRange(`AI${row}:AJ${row}`).formulas = [[`=--(AH${row}>=0.5)`, `=--(AI${row}=O${row})`]];
  targetSheet.getRange(`AL${row}:AM${row}`).formulas = [[`=--(AK${row}>=0.5)`, `=--(AL${row}=O${row})`]];
}
targetSheet.getRange(`AH2:AH${targets.length + 1}`).format.numberFormat = "0.0000";
targetSheet.getRange(`AK2:AK${targets.length + 1}`).format.numberFormat = "0.0000";
targetSheet.getRange(`A2:AM${targets.length + 1}`).format.font = { name: "Arial", size: 9 };
targetSheet.getRange(`M2:M${targets.length + 1}`).format.wrapText = true;
targetSheet.getRange(`R2:S${targets.length + 1}`).format.wrapText = true;
targetSheet.getRange(`AG2:AG${targets.length + 1}`).format.wrapText = true;
targetSheet.getRange(`A2:AM${targets.length + 1}`).format.rowHeight = 32;
targetSheet.getRange(`Q2:Q${targets.length + 1}`).conditionalFormats.add("containsText", { text: "PRESENT", format: { fill: colors.greenLight, font: { color: colors.green, bold: true } } });
targetSheet.getRange(`Q2:Q${targets.length + 1}`).conditionalFormats.add("containsText", { text: "ABSTAIN", format: { fill: colors.amberLight, font: { color: colors.amber } } });
targetSheet.getRange(`V2:V${targets.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: 0, format: { fill: colors.coralLight, font: { color: colors.coral } } });
targetSheet.tables.add(`A1:AM${targets.length + 1}`, true, "MixedLabelTargets").style = "TableStyleMedium2";
targetSheet.freezePanes.freezeRows(1);
targetSheet.freezePanes.freezeColumns(4);
const targetWidths = {
  A: 10, B: 11, C: 12, D: 36, E: 10, F: 10, G: 20, H: 9, I: 9, J: 9, K: 9, L: 8,
  M: 32, N: 14, O: 10, P: 14, Q: 22, R: 26, S: 26, T: 12, U: 16, V: 10, W: 10,
  X: 10, Y: 20, Z: 12, AA: 13, AB: 14, AC: 18, AD: 18, AE: 14, AF: 14, AG: 62,
  AH: 13, AI: 13, AJ: 13, AK: 15, AL: 15, AM: 15,
};
for (const [column, width] of Object.entries(targetWidths)) targetSheet.getRange(`${column}1:${column}${targets.length + 1}`).format.columnWidth = width;

const paperHeaders = ["REFNO", "目标数", "历史R3=0", "历史R3=1", "历史ROCKMAG非空", "预测非空", "明确判定", "拒判", "ROCKMAG存在正确", "直接R3正确", "原基线正确", "成对排序正确", "全目标R3正确", "恒定预测"];
paperSheet.getRange("A1:N1").values = [paperHeaders];
headers(paperSheet.getRange("A1:N1"));
paperSheet.getRange(`A2:N${papers.length + 1}`).values = papers.map((row) => [
  row.REFNO, row.targets, row.historical_r3_0, row.historical_r3_1, row.historical_rockmag_nonempty, row.predicted_present,
  row.decided, row.abstained, row.rockmag_presence_correct, row.direct_r3_correct, row.old_baseline_correct, row.old_pairwise_correct,
  row.all_direct_r3_correct, row.constant_direct_prediction,
]);
paperSheet.tables.add(`A1:N${papers.length + 1}`, true, "MixedLabelPapers").style = "TableStyleMedium2";
paperSheet.getRange(`A2:N${papers.length + 1}`).format.font = { name: "Arial", size: 9 };
paperSheet.getRange(`A1:N${papers.length + 1}`).format.columnWidth = 14;
paperSheet.getRange(`A1:A${papers.length + 1}`).format.columnWidth = 11;
paperSheet.getRange(`E1:E${papers.length + 1}`).format.columnWidth = 19;
paperSheet.getRange(`I1:L${papers.length + 1}`).format.columnWidth = 18;
paperSheet.getRange(`M2:N${papers.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: 1, format: { fill: colors.greenLight, font: { color: colors.green } } });
paperSheet.freezePanes.freezeRows(1);

const mismatches = targets.filter((row) => row.rockmag_presence_equals_r3 === 0);
mismatchSheet.getRange("A1:F1").values = [["REFNO", "RESULTNO", "ROCKNAME", "历史 ROCKMAG", "ROCKMAG 非空", "历史 R3"]];
headers(mismatchSheet.getRange("A1:F1"));
mismatchSheet.getRange(`A2:F${mismatches.length + 1}`).values = mismatches.map((row) => [
  row.REFNO, row.RESULTNO, row.ROCKNAME, row.historical_rockmag, row.truth_rockmag_present, row.historical_r3,
]);
mismatchSheet.tables.add(`A1:F${mismatches.length + 1}`, true, "RockmagR3Mismatches").style = "TableStyleMedium2";
mismatchSheet.getRange(`A1:A${mismatches.length + 1}`).format.columnWidth = 11;
mismatchSheet.getRange(`B1:B${mismatches.length + 1}`).format.columnWidth = 12;
mismatchSheet.getRange(`C1:C${mismatches.length + 1}`).format.columnWidth = 42;
mismatchSheet.getRange(`D1:D${mismatches.length + 1}`).format.columnWidth = 38;
mismatchSheet.getRange(`E1:F${mismatches.length + 1}`).format.columnWidth = 15;
mismatchSheet.getRange(`D2:D${mismatches.length + 1}`).format.wrapText = true;
mismatchSheet.getRange(`A2:F${mismatches.length + 1}`).format.rowHeight = 30;
mismatchSheet.freezePanes.freezeRows(1);

title(protocolSheet, protocolSheet.getRange("A1:C1"), "测试口径与可复现性说明");
protocolSheet.getRange("A3:C3").values = [["项目", "内容", "审计值"]];
section(protocolSheet.getRange("A3:C3"));
const protocolRows = [
  ["测试集定义", "同一 REFNO 至少两个目标，且历史 R3 同时有 0 和 1", "61篇 / 278目标"],
  ["正文来源", "冻结页面缓存，统一重建逐页文本", "61/61篇，1,135页"],
  ["预测器可见字段", "REFNO、RESULTNO、ROCKUNITNO、ROCKNAME、RLAT/RLONG、COMPONENT、SLAT/SLONG、DEC/INC、PLAT/PLONG、PTYPE", "14字段"],
  ["预测器禁止来源", "PMAGRESULT.ROCKMAG/TESTS/LABDETAILS/COMMENTS、历史R3、旧预测、旧评测", "均未读取"],
  ["PRESENT", "找到足够的目标绑定方法与载磁矿物或诊断证据，输出非空 ROCKMAG", "122行"],
  ["ABSENT", "目标锚点可靠但未发现 ROCKMAG 证据，输出明确空值", "12行"],
  ["ABSTAIN", "目标锚点或证据不足，不应解释为真实空值", "144行"],
  ["全量强制二分类", "沿用现有 direct_r3 规则：只有 PRESENT=1，其余状态=0", "仅用于与原表同分母比较"],
  ["明确判定口径", "仅统计 PRESENT 与 ABSENT", "134/278 = 48.20%"],
  ["标签关系", "历史 ROCKMAG 是否非空并不完全等同历史 R3", "18行不一致；一致率93.53%"],
  ["字段值评估", "ROCKMAG 存在性是主指标；字符串完全一致另列，不把编码同义词自动视为相同", "真非空且预测非空78行，仅4行字符串完全一致"],
  ["身份清单 SHA-256", summary.artifacts.identity_manifest, summary.artifacts.identity_manifest_sha256],
  ["盲测预测 SHA-256", summary.artifacts.predictions, summary.artifacts.predictions_sha256],
];
protocolSheet.getRange(`A4:C${protocolRows.length + 3}`).values = protocolRows;
protocolSheet.getRange(`A4:C${protocolRows.length + 3}`).format = { font: { name: "Arial", size: 10 }, wrapText: true, verticalAlignment: "top" };
protocolSheet.getRange(`A4:C${protocolRows.length + 3}`).format.rowHeight = 43;
protocolSheet.getRange(`A3:C${protocolRows.length + 3}`).format.borders = { preset: "inside", style: "thin", color: colors.gray };
protocolSheet.getRange(`A1:A${protocolRows.length + 3}`).format.columnWidth = 24;
protocolSheet.getRange(`B1:B${protocolRows.length + 3}`).format.columnWidth = 78;
protocolSheet.getRange(`C1:C${protocolRows.length + 3}`).format.columnWidth = 48;
protocolSheet.freezePanes.freezeRows(3);

const inspectSummary = await workbook.inspect({ kind: "table", range: "结果摘要!A1:D24", include: "values,formulas", tableMaxRows: 30, tableMaxCols: 8 });
const inspectTargets = await workbook.inspect({ kind: "table", range: "目标明细!A1:AM8", include: "values,formulas", tableMaxRows: 8, tableMaxCols: 39 });
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
await fs.writeFile(path.join(outputDir, "report_inspect.ndjson"), `${inspectSummary.ndjson}\n${inspectTargets.ndjson}\n${errors.ndjson}\n`);

const previews = [
  ["结果摘要", "A1:D24", "preview_summary.png"],
  ["目标明细", "A1:W16", "preview_targets_core.png"],
  ["目标明细", "X1:AM16", "preview_targets_evidence.png"],
  ["论文汇总", "A1:N20", "preview_papers.png"],
  ["ROCKMAG-R3不一致", `A1:F${mismatches.length + 1}`, "preview_mismatches.png"],
  ["口径与溯源", `A1:C${protocolRows.length + 3}`, "preview_protocol.png"],
];
for (const [sheetName, range, file] of previews) {
  const blob = await workbook.render({ sheetName, range, scale: 1.25, format: "png" });
  await fs.writeFile(path.join(outputDir, file), new Uint8Array(await blob.arrayBuffer()));
}

let output;
try {
  output = await SpreadsheetFile.exportXlsx(workbook);
} catch (error) {
  console.error("EXPORT_ERROR", error?.message, error?.name, error?.stack?.split("\n").slice(0, 8).join("\n"));
  process.exitCode = 2;
  throw error;
}
await output.save(path.join(outputDir, "rockmag_mixed_label_multitarget_test_report.xlsx"));
const reopened = await SpreadsheetFile.importXlsx(
  await FileBlob.load(path.join(outputDir, "rockmag_mixed_label_multitarget_test_report.xlsx")),
);
const finalSummary = await reopened.inspect({ kind: "table", range: "结果摘要!A1:D24", include: "values,formulas", tableMaxRows: 30, tableMaxCols: 8 });
const finalErrors = await reopened.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "reopened export formula error scan" });
await fs.writeFile(path.join(outputDir, "final_export_inspect.ndjson"), `${finalSummary.ndjson}\n${finalErrors.ndjson}\n`);
const finalPreview = await reopened.render({ sheetName: "结果摘要", range: "A1:D24", scale: 1.25, format: "png" });
await fs.writeFile(path.join(outputDir, "preview_final_summary.png"), new Uint8Array(await finalPreview.arrayBuffer()));
console.log(JSON.stringify({ output: path.join(outputDir, "rockmag_mixed_label_multitarget_test_report.xlsx"), sheets: 5, targets: targets.length, papers: papers.length, mismatches: mismatches.length }, null, 2));
