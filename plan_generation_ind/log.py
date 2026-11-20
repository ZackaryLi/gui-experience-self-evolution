import os
from datetime import datetime
from pathlib import Path


def log(msg: str, log_file: str|Path="run.log", append: bool = True):
    # print(msg)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    mode = "a" if append else "w"
    if os.path.exists(log_file):
        try:
            if not append:
                with open(log_file, mode, encoding="utf-8") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n")
                    f.write(str(msg) + "\n")
            else:
                with open(log_file, mode, encoding="utf-8") as f:
                    f.write(str(msg) + "\n")
        except Exception as e:
            print(f"[日志写入失败] {e}")
    else:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")
        except Exception as e:
            print(f"[日志写入失败] {e}")