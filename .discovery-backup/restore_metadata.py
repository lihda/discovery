from pathlib import Path
import json, os
root=Path(__file__).resolve().parent.parent
manifest=json.loads((Path(__file__).parent/'manifest.json').read_text())
for item in manifest['entries']:
    path=root/item['path']
    if item['type']=='directory': path.mkdir(parents=True,exist_ok=True)
for item in reversed(manifest['entries']):
    path=root/item['path']
    if item['type'] in ('file','directory'):
        path.chmod(int(item['mode'],8))
        os.utime(path,ns=(item['mtime_ns'],item['mtime_ns']),follow_symlinks=False)
print('Directory, permission and modification-time metadata restored.')
