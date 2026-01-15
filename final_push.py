import subprocess
import os

def final_sync():
    # 1. 彻底清除代理，防止干扰连接
    os.system("git config --global --unset http.proxy")
    os.system("git config --global --unset https.proxy")

    # 2. 强制重置 Git 追踪（这会移除主页上所有已删除文件的记录）
    commands = [
        "git rm -r --cached .",
        "git add bert_demo.py README.md requirements.txt", # 只添加这三个核心文件
        "git add .gitignore",
        'git commit -m "refactor: final streamlined product version"',
        "git branch -M main",
        "git push -u origin main --force" # 强制推送以覆盖冗余历史
    ]

    for cmd in commands:
        print(f"Executing: {cmd}")
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    final_sync()