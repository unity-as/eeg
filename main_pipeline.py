import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "scripts/step1_build_dataset.py",
    "scripts/step2_train.py",
]

cfg = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"

for script in scripts:
    print(f"\n===== 运行 {script} =====")
    cmd = [sys.executable, script, cfg]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"脚本 {script} 运行失败，停止。")
        sys.exit(result.returncode)

print("全部流程完成！")
