import kagglehub
from pathlib import Path
import shutil
import os

path = kagglehub.dataset_download("cymerunaslam/allthenews")
print("Path to dataset files:", path)

dest = Path("data") / "allthenews"
dest.mkdir(parents=True, exist_ok=True)

src = Path(path)

if not src.exists():
    raise FileNotFoundError(f"Returned path does not exist: {src}")

if src.is_dir():
    for item in src.iterdir():
        if item.is_file():
            target = dest / item.name
            if target.exists():
                target.unlink()
            try:
                shutil.copy2(item, target)   # copy metadata too
                item.unlink()                # remove original
                print(f"Copied and removed {item} -> {target}")
            except Exception as e:
                print(f"Failed to move {item} -> {target}: {e}")

elif src.is_file():
    target = dest / src.name
    if target.exists():
        target.unlink()
    try:
        shutil.copy2(src, target)
        src.unlink()
        print(f"Copied and removed {src} -> {target}")
    except Exception as e:
        print(f"Failed to move {src} -> {target}: {e}")

else:
    raise FileNotFoundError(f"Returned path is neither file nor directory: {src}")
