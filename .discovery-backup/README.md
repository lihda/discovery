# Discovery folder backup

This branch contains the complete local discovery folder alongside the three files already present on the original main branch. The API key in paleomagnetic-r3-workflow/dist/paleomagnetic-r1-r7-main-workflow/.env is intentionally empty; the original .env is saved separately on the original computer.

Files over 50 MiB use Git LFS. Clone this branch with Git LFS installed and run `git lfs pull` to obtain their full contents. `manifest.json` records every original file, directory and symbolic link; the sanitized .env hash is recorded instead of its original hash. Git does not preserve empty directories or all file permissions. Run `python3 .discovery-backup/restore_metadata.py` after checkout to recreate directories and restore permission modes and modification times.

Some symbolic links point to Python and dependency installations outside this project on the original computer. Their link targets are preserved, but those external installations are not part of this folder backup.

The main branch is unchanged. Upload completion and downloaded-content verification are recorded separately by the uploader.
