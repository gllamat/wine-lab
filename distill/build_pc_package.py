"""Builds distill/wset_distill_pc.zip: the folder to copy to the Windows PC (notebook + data pack + setup/run scripts + app).
Run after rebuilding the notebook or the pack:  python distill/build_pc_package.py"""
import shutil, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, PACK, NB = HERE / "pc_package_src", HERE / "pack", HERE / "wset_distill.ipynb"
OUT = HERE / "wset_distill_pc"

shutil.rmtree(OUT, ignore_errors=True); (OUT / "pack").mkdir(parents=True)
for f in SRC.iterdir():
    if f.is_dir():
        shutil.copytree(f, OUT / f.name, ignore=shutil.ignore_patterns('__pycache__')); continue
    data = f.read_bytes()
    if f.suffix == ".bat":                                     # Windows batch files need CRLF line endings
        data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    (OUT / f.name).write_bytes(data)
shutil.copy(NB, OUT / NB.name)
for f in ("wset_distill_pack.parquet", "wset_distill_schema.json"):
    shutil.copy(PACK / f, OUT / "pack" / f)

zpath = HERE / "wset_distill_pc.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob("*")):
        if f.is_file(): z.write(f, Path("wset_distill_pc") / f.relative_to(OUT))
print("wrote", zpath, f"({zpath.stat().st_size / 1e6:.1f} MB)")
