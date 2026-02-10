# M:\meeting\diagnose_speechbrain.py

import sys
import os
import subprocess
import platform

print("--- 诊断开始 ---")
print(f"当前工作目录: {os.getcwd()}")
print(f"Python 解释器路径 (sys.executable): {sys.executable}")
print(f"Python 版本: {sys.version}")

print("\n--- 'where python' (或 'which python') 命令输出 ---")
try:
    if platform.system() == "Windows":
        result = subprocess.run(['where', 'python'], capture_output=True, text=True, check=True)
    else: # macOS/Linux
        result = subprocess.run(['which', 'python'], capture_output=True, text=True, check=True)
    print(result.stdout)
except Exception as e:
    print(f"执行 where/which python 命令失败: {e}")


print("\n--- sys.path (Python 模块搜索路径) ---")
for p in sys.path:
    print(p)
print("--------------------------------------")


print("\n--- 尝试导入 speechbrain 模块 ---")
try:
    import speechbrain
    print(f"✅ speechbrain 模块导入成功!")
    print(f"   speechbrain 安装位置: {speechbrain.__file__}")
    print(f"   speechbrain 版本: {speechbrain.__version__}")
except Exception as e:
    print(f"❌ 导入 speechbrain 模块失败: {e}")
print("---------------------------------")


print("\n--- 尝试导入 speechbrain.hyperparams 子模块 ---")
try:
    import speechbrain.hyperparams
    print(f"✅ speechbrain.hyperparams 子模块导入成功!")
    print(f"   hyperparams 模块位置: {speechbrain.hyperparams.__file__}")
except Exception as e:
    print(f"❌ 导入 speechbrain.hyperparams 子模块失败: {e}")
print("-----------------------------------------------")

print("\n--- 诊断结束 ---")