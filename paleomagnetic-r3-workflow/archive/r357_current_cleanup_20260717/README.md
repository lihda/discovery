# R3/R5/R7 2026-07-17 Cleanup Archive

Source project:

`/Users/lihd/discovery/paleomagnetic-r3-workflow/dist/paleomagnetic-r357-calibrated-workflow-full`

This archive contains files removed from the active project during the
2026-07-17 cleanup. Nothing here is part of the retained current-best runtime.

Archived groups:

- the 2026-07-11 business output and old strict scoring output;
- entity-router, layout and NLI experiments that were not promoted;
- historical selective/practice-criterion code and artifacts;
- the MIL evaluator output, while the label-blind MIL bag builder remains active;
- trace feature versions 1-3; version 4 remains active;
- the superseded paleomagnetic router copy from the mixed output directory;
- patent drafts and historical baseline notes not required by the current run.

`MOVE_LIST.txt` records the original relative paths. `SHA256SUMS.txt` covers
every other regular file in this archive. Verify with:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

The active project keeps hard links to two provenance inputs:

- `outputs/r357_latest_full_07111019/r3_page_source_cache.jsonl`;
- four low-priority page-cache fallbacks already stored in the 2026-07-13
  legacy archive.

Those hard links preserve frozen audit paths and hashes without duplicating
physical disk blocks. Do not remove them unless the downstream hash chain is
formally re-frozen and recomputed.
