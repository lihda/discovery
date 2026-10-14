import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workDir = new URL(".", import.meta.url).pathname;
const outputDir = new URL("../", import.meta.url).pathname;
const report = JSON.parse(await fs.readFile(`${workDir}/accuracy_data.json`, "utf8"));
const rows = report.rows;
const lastRow = rows.length + 1;

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const comparison = workbook.worksheets.add("Comparison");
const evidence = workbook.worksheets.add("Evidence");
for (const sheet of [summary, comparison, evidence]) sheet.showGridLines = false;

summary.mergeCells("A1:L1");
summary.getRange("A1").values = [["R3/R5/R7 LLM Accuracy - Native Unmapped Targets"]];
summary.getRange("A1:L1").format = {
  fill: "#25313C",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:L1").format.rowHeight = 34;
const summaryHeaders = [
  "Indicator", "Total", "Correct", "Accuracy", "Unknown", "Decided",
  "Decided Correct", "Decided Accuracy", "TP", "TN", "FP", "FN",
];
summary.getRange("A3:L3").values = [summaryHeaders];
summary.getRange("A4:A6").values = [["R3"], ["R5"], ["R7"]];
const comparisonColumns = {
  R3: { gold: "D", pred: "E", correct: "F", outcome: "G" },
  R5: { gold: "H", pred: "I", correct: "J", outcome: "K" },
  R7: { gold: "L", pred: "M", correct: "N", outcome: "O" },
};
for (const [index, indicator] of ["R3", "R5", "R7"].entries()) {
  const row = index + 4;
  const cols = comparisonColumns[indicator];
  summary.getRange(`B${row}:L${row}`).formulas = [[
    `=COUNTA('Comparison'!$A$2:$A$${lastRow})`,
    `=SUM('Comparison'!$${cols.correct}$2:$${cols.correct}$${lastRow})`,
    `=C${row}/B${row}`,
    `=COUNTIF('Comparison'!$${cols.outcome}$2:$${cols.outcome}$${lastRow},"UNKNOWN")`,
    `=B${row}-E${row}`,
    `=COUNTIFS('Comparison'!$${cols.outcome}$2:$${cols.outcome}$${lastRow},"<>UNKNOWN",'Comparison'!$${cols.correct}$2:$${cols.correct}$${lastRow},1)`,
    `=IF(F${row}=0,"",G${row}/F${row})`,
    `=COUNTIFS('Comparison'!$${cols.gold}$2:$${cols.gold}$${lastRow},1,'Comparison'!$${cols.pred}$2:$${cols.pred}$${lastRow},1)`,
    `=COUNTIFS('Comparison'!$${cols.gold}$2:$${cols.gold}$${lastRow},0,'Comparison'!$${cols.pred}$2:$${cols.pred}$${lastRow},0)`,
    `=COUNTIFS('Comparison'!$${cols.gold}$2:$${cols.gold}$${lastRow},0,'Comparison'!$${cols.pred}$2:$${cols.pred}$${lastRow},1)`,
    `=COUNTIFS('Comparison'!$${cols.gold}$2:$${cols.gold}$${lastRow},1,'Comparison'!$${cols.pred}$2:$${cols.pred}$${lastRow},0)`,
  ]];
}
summary.getRange("A3:L3").format = {
  fill: "#385568",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
summary.getRange("A4:L6").format.borders = { preset: "inside", style: "thin", color: "#D6DEE3" };
summary.getRange("D4:D6").format.numberFormat = "0.0%";
summary.getRange("H4:H6").format.numberFormat = "0.0%";
summary.getRange("A8:L8").merge();
summary.getRange("A8").values = [[
  "Test design: 10 original RESULTNO values outside all three 2,800-entry maps; 30/30 API calls returned unique response IDs.",
]];
summary.getRange("A9:L9").merge();
summary.getRange("A9").values = [[
  "Accuracy compares final compatibility scores with targets.xlsx; UNKNOWN maps to R3=0 and R5/R7=1.",
]];
summary.getRange("A8:L9").format = {
  fill: "#EEF3F5",
  font: { color: "#33444D" },
  wrapText: true,
  verticalAlignment: "center",
};
summary.getRange("A8:L9").format.rowHeight = 26;
summary.freezePanes.freezeRows(3);
summary.getRange("A1:A9").format.columnWidth = 15;
summary.getRange("B1:L9").format.columnWidth = 14;

const comparisonHeaders = [
  "RESULTNO", "REFNO", "ROCKNAME",
  "R3_GOLD", "R3_PRED", "R3_CORRECT", "R3_OUTCOME",
  "R5_GOLD", "R5_PRED", "R5_CORRECT", "R5_OUTCOME",
  "R7_GOLD", "R7_PRED", "R7_CORRECT", "R7_OUTCOME",
];
comparison.getRange(`A1:O${lastRow}`).values = [
  comparisonHeaders,
  ...rows.map((row) => [
    row.RESULTNO, row.REFNO, row.ROCKNAME,
    row.R3_GOLD, row.R3_PRED, null, row.R3_OUTCOME,
    row.R5_GOLD, row.R5_PRED, null, row.R5_OUTCOME,
    row.R7_GOLD, row.R7_PRED, null, row.R7_OUTCOME,
  ]),
];
comparison.getRange("F2").formulas = [["=--(D2=E2)"]];
comparison.getRange(`F2:F${lastRow}`).fillDown();
comparison.getRange("J2").formulas = [["=--(H2=I2)"]];
comparison.getRange(`J2:J${lastRow}`).fillDown();
comparison.getRange("N2").formulas = [["=--(L2=M2)"]];
comparison.getRange(`N2:N${lastRow}`).fillDown();
comparison.freezePanes.freezeRows(1);
comparison.freezePanes.freezeColumns(3);
comparison.getRange("A1:O1").format = {
  fill: "#25313C",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
comparison.getRange("A1:O1").format.rowHeight = 34;
comparison.getRange(`A2:C${lastRow}`).format.numberFormat = "@";
comparison.getRange(`A2:O${lastRow}`).format.verticalAlignment = "center";
comparison.getRange(`A2:A${lastRow}`).format.columnWidth = 13;
comparison.getRange(`B2:B${lastRow}`).format.columnWidth = 11;
comparison.getRange(`C2:C${lastRow}`).format.columnWidth = 34;
comparison.getRange(`D2:O${lastRow}`).format.columnWidth = 13;
const comparisonTable = comparison.tables.add(`A1:O${lastRow}`, true, "AccuracyComparison");
comparisonTable.style = "TableStyleMedium2";
for (const column of ["F", "J", "N"]) {
  const range = comparison.getRange(`${column}2:${column}${lastRow}`);
  range.conditionalFormats.add("cellIs", {
    operator: "equal",
    formula: 1,
    format: { fill: "#DDEFE5", font: { color: "#17623A", bold: true } },
  });
  range.conditionalFormats.add("cellIs", {
    operator: "equal",
    formula: 0,
    format: { fill: "#F8DEDE", font: { color: "#982C2C", bold: true } },
  });
}

const evidenceHeaders = [
  "RESULTNO", "REFNO", "ROCKNAME", "INDICATOR", "GOLD", "PRED", "CORRECT",
  "OUTCOME", "STATUS", "REASONING", "EXACT_QUOTE", "RESPONSE_ID",
];
const evidenceRows = [];
for (const row of rows) {
  for (const indicator of ["R3", "R5", "R7"]) {
    evidenceRows.push([
      row.RESULTNO,
      row.REFNO,
      row.ROCKNAME,
      indicator,
      row[`${indicator}_GOLD`],
      row[`${indicator}_PRED`],
      row[`${indicator}_CORRECT`],
      row[`${indicator}_OUTCOME`],
      row[`${indicator}_STATUS`],
      row[`${indicator}_REASONING`],
      row[`${indicator}_QUOTE`],
      row[`${indicator}_RESPONSE_ID`],
    ]);
  }
}
const evidenceLastRow = evidenceRows.length + 1;
evidence.getRange(`A1:L${evidenceLastRow}`).values = [evidenceHeaders, ...evidenceRows];
evidence.freezePanes.freezeRows(1);
evidence.freezePanes.freezeColumns(4);
evidence.getRange("A1:L1").format = {
  fill: "#25313C",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
evidence.getRange("A1:L1").format.rowHeight = 34;
evidence.getRange(`A2:D${evidenceLastRow}`).format.numberFormat = "@";
evidence.getRange(`A2:L${evidenceLastRow}`).format.verticalAlignment = "top";
evidence.getRange(`J2:K${evidenceLastRow}`).format.wrapText = true;
evidence.getRange(`A2:L${evidenceLastRow}`).format.rowHeight = 54;
evidence.getRange(`A2:A${evidenceLastRow}`).format.columnWidth = 13;
evidence.getRange(`B2:B${evidenceLastRow}`).format.columnWidth = 11;
evidence.getRange(`C2:C${evidenceLastRow}`).format.columnWidth = 30;
evidence.getRange(`D2:I${evidenceLastRow}`).format.columnWidth = 12;
evidence.getRange(`J2:K${evidenceLastRow}`).format.columnWidth = 54;
evidence.getRange(`L2:L${evidenceLastRow}`).format.columnWidth = 42;
const evidenceTable = evidence.tables.add(`A1:L${evidenceLastRow}`, true, "ApiEvidenceAudit");
evidenceTable.style = "TableStyleMedium4";

await fs.mkdir(outputDir, { recursive: true });
for (const [sheetName, range, fileName] of [
  ["Summary", "A1:L9", "accuracy_summary_preview.png"],
  ["Comparison", `A1:O${lastRow}`, "accuracy_comparison_preview.png"],
  ["Evidence", `A1:L${evidenceLastRow}`, "accuracy_evidence_preview.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${workDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

console.log((await workbook.inspect({
  kind: "table",
  range: "Summary!A1:L9",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 12,
  maxChars: 12000,
})).ndjson);
console.log((await workbook.inspect({
  kind: "table",
  range: `Comparison!A1:O${lastRow}`,
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 15,
  maxChars: 16000,
})).ndjson);
console.log((await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
})).ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/r357_accuracy.xlsx`);
