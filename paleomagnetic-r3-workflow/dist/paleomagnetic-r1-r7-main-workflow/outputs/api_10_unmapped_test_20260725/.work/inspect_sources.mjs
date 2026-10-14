import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const paths = [
  "/Users/lihd/discovery/paleomagnetic-r3-workflow/dist/paleomagnetic-r357-calibrated-workflow-full/data/targets.xlsx",
  "/Users/lihd/discovery/paleomagnetic-r3-workflow/dist/paleomagnetic-r357-calibrated-workflow-full/data/source_metadata/PMAGRESULT.xlsx",
  "/Users/lihd/discovery/paleomagnetic-r3-workflow/dist/paleomagnetic-r357-calibrated-workflow-full/data/source_metadata/ROCKUNIT.xlsx",
  "/Users/lihd/discovery/paleomagnetic-r3-workflow/dist/paleomagnetic-r357-calibrated-workflow-full/outputs/r357_paper_only_eval_07111638/r5_latest_source_recovered_predictions.xlsx",
  "/Users/lihd/discovery/paleomagnetic-r3-workflow/dist/paleomagnetic-r357-calibrated-workflow-full/outputs/r357_paper_only_eval_07111638/r7_latest_source_recovered_predictions.xlsx",
];

for (const path of paths) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const inspection = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 16000,
    tableMaxRows: 5,
    tableMaxCols: 50,
    tableMaxCellChars: 180,
  });
  console.log(`SOURCE ${path}`);
  console.log(inspection.ndjson);
}
