import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workDir = new URL(".", import.meta.url).pathname;
const outputDir = new URL("../", import.meta.url).pathname;
const selection = JSON.parse(await fs.readFile(`${workDir}/selection.json`, "utf8"));
const headers = [
  "REFNO",
  "RESULTNO",
  "ROCKUNITNO",
  "ROCKNAME",
  "HIMAGAGE",
  "LOMAGAGE",
  "HIGHAGE",
  "LOWAGE",
  "B",
  "N",
  "KD",
  "ED95",
  "DEMAGCODE",
  "TESTS",
  "R3_EVIDENCE_TEXT",
  "R3_EVIDENCE_SOURCE",
  "R5_EVIDENCE_TEXT",
  "R5_EVIDENCE_SOURCE",
  "R7_EVIDENCE_TEXT",
  "R7_EVIDENCE_SOURCE",
];
const values = [headers, ...selection.input_rows.map((row) => headers.map((header) => row[header] ?? null))];

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Input");
sheet.showGridLines = false;
sheet.getRange(`A1:T${values.length}`).values = values;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(4);

sheet.getRange("A1:T1").format = {
  fill: "#25313C",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: "#25313C" },
};
sheet.getRange("A1:T1").format.rowHeight = 36;
sheet.getRange(`A2:N${values.length}`).format.verticalAlignment = "top";
sheet.getRange(`O2:T${values.length}`).format = {
  verticalAlignment: "top",
  wrapText: true,
};
sheet.getRange(`A2:D${values.length}`).format.numberFormat = "@";
sheet.getRange(`E2:M${values.length}`).format.numberFormat = "0.##";
sheet.getRange(`A2:T${values.length}`).format.rowHeight = 90;

const widths = {
  A: 12,
  B: 14,
  C: 14,
  D: 32,
  E: 12,
  F: 12,
  G: 12,
  H: 12,
  I: 10,
  J: 10,
  K: 10,
  L: 10,
  M: 12,
  N: 12,
  O: 56,
  P: 34,
  Q: 56,
  R: 34,
  S: 56,
  T: 34,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}1:${column}${values.length}`).format.columnWidth = width;
}

const table = sheet.tables.add(`A1:T${values.length}`, true, "UnmappedTargetsInput");
table.style = "TableStyleMedium2";
table.showBandedRows = true;
table.showFilterButton = true;

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({
  sheetName: "Input",
  range: `A1:T${values.length}`,
  scale: 0.8,
  format: "png",
});
await fs.writeFile(`${workDir}/input_preview.png`, new Uint8Array(await preview.arrayBuffer()));

const inspection = await workbook.inspect({
  kind: "table",
  range: `Input!A1:T${values.length}`,
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 20,
  tableMaxCellChars: 120,
  maxChars: 14000,
});
console.log(inspection.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/input.xlsx`);
