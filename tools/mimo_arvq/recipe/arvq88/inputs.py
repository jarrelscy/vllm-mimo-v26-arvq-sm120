import json
from pathlib import Path
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
