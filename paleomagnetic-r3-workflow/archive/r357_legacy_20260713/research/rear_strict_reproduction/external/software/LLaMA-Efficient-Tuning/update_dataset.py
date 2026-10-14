dic = {
  "26doc-1523disd": {
    "file_name": "26doc-1523disd.json",
  },
  "26doc-1523dis": {
    "file_name": "26doc-1523dis.json",
  },
  "26doc-1523pos": {
    "file_name": "26doc-1523pos.json",
  },
  "26doc-1523cot": {
    "file_name": "26doc-1523cot.json",
  },
  "26doc-152311": {
    "file_name": "26doc-152311.json",
  },
  "26doc-152301": {
    "file_name": "26doc-152301.json",
  },
  "26doc-152303": {
    "file_name": "26doc-152303.json",
  },
  "26doc-152305": {
    "file_name": "26doc-152305.json",
  },
  "26doc-15231": {
    "file_name": "26doc-15231.json",
  },
  "26doc-15153": {
    "file_name": "26doc-15153.json",
  },
  "26doc-15151": {
    "file_name": "26doc-15151.json",
  },
  "26doc-15235": {
    "file_name": "26doc-15235.json",
  },
  "26doc-15233": {
    "file_name": "26doc-15233.json",
  },
  "26doc-15233-new": {
    "file_name": "26doc-15233-new.json",
  },
  "26doc-1523": {
    "file_name": "26doc-1523.json",
  },
  "26doc-1523": {
    "file_name": "26doc-1523.json",
  },
  "26doc-1533": {
    "file_name": "26doc-1533.json",
  },
  "26doc-1526": {
    "file_name": "26doc-1526.json",
  },
  "26doc-313": {
    "file_name": "26doc-313.json",
  },
  "ungrouped-214-a10-f": {
    "file_name": "ungrouped-214-a10-f.json",
  },
  "ungrouped-26-a0-f": {
    "file_name": "ungrouped-26-a0-f.json",
  },
  "ungrouped-26-a1-f": {
    "file_name": "ungrouped-26-a1-f.json",
  },
  "ungrouped-26-a2-f": {
    "file_name": "ungrouped-26-a2-f.json",
  },
  "ungrouped-26-a5-f": {
    "file_name": "ungrouped-26-a5-f.json",
  },
  "ungrouped-26-a10-f": {
    "file_name": "ungrouped-26-a10-f.json"
  },
  "ungrouped-26-a15-f": {
    "file_name": "ungrouped-26-a15-f.json"
  },
  "ungrouped-26-ainf-f": {
    "file_name": "ungrouped-26-ainf-f.json"
  },
  "grouped-2-f": {
    "file_name": "grouped-2-f.json",
  },
  "grouped-22-f": {
    "file_name": "grouped-22-f.json",
  },
  "grouped-13-f": {
    "file_name": "grouped-13-f.json",
  },
  "ungrouped-2": {
    "file_name": "ungrouped-2.json",
  },
  "grouped-2": {
    "file_name": "grouped-2.json",
  },
  "grouped-22": {
    "file_name": "grouped-22.json",
  },
  "grouped-13": {
    "file_name": "grouped-13.json",
  },
  "ranksft-list": {
    "file_name": "ranksft-list.json",
  },
  "ranksft-gold": {
    "file_name": "ranksft-gold.json",
  },
  "ranksft-equal": {
    "file_name": "ranksft-equal.json",
  },
  "ranksft-unbias": {
    "file_name": "ranksft-unbias.json",
  },
  "ranksft-re": {
    "file_name": "ranksft-re.json",
  },
  "ranksft-hard": {
    "file_name": "ranksft-hard.json",
  },
  "ranksft-hard-2": {
    "file_name": "ranksft-hard-2.json",
  },
  "ranksft": {
    "file_name": "ranksft.json",
  },
  "rro-rl-qa-v2-force": {
    "file_name": "nq-train-rl-score-qa-v2-force.json",
    "ranking": True
  },
  "rl-qa-v2-force":{
    "file_name": "nq-train-rl-qa-v2-force.json",
    "ranking": True
  },
  "rl-qa-v2-force-100":{
    "file_name": "nq-train-rl-qa-v2-force-100.json"
  },
  "nofix": {
    "file_name": "nq-train-pure-qa-dense-nofix.json"
  },
  "sel-col-force": {
    "file_name": "nq-train-pure-qa-dense-sel-col-force.json"
  },
  "qa-col-force": {
    "file_name": "nq-train-pure-qa-dense-qa-col-force.json"
  },
  "qa-v2-force": {
    "file_name": "nq-train-pure-qa-dense-qa-v2-force.json"
  },
  "sel-v2-force": {
    "file_name": "nq-train-pure-qa-dense-sel-v2-force.json"
  },
  "sel-nofix": {
    "file_name": "nq-train-pure-qa-dense-sel-nofix.json"
  },
  "pos1-focus": {
    "file_name": "nq-train-pure-qa-dense-pos1-focus.json"
  },
  "pos1-unfocus": {
    "file_name": "nq-train-pure-qa-dense-pos1-unfocus.json"
  },
  "mixfocus": {
    "file_name": "nq-train-pure-qa-dense-mixfocus.json"
  },
  "mixfocus2": {
    "file_name": "nq-train-pure-qa-dense-mixfocus2.json"
  },
  "select": {
    "file_name": "nq-train-pure-qa-dense-select.json"
  },
  "select-mul": {
    "file_name": "nq-train-pure-qa-dense-select-mul.json"
  }
}

import json

json.dump(dic, open("../data/training/dataset_info.json", "w", encoding='utf-8'), ensure_ascii=False, indent=4)