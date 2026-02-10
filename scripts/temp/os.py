import os
from huggingface_hub import hf_hub_download, list_repo_files
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量，包括你的 HUGGINGFACE_AUTH_TOKEN
load_dotenv()
HF_AUTH_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")

# 如果你的令牌没有设置，或者你不确定，可以使用提示输入
if not HF_AUTH_TOKEN:
    print("HUGGINGFACE_AUTH_TOKEN 环境变量未设置。请确保你的 .env 文件中有它，")
    print("或者你可以在运行此脚本之前手动设置环境变量。")
    # 如果实在没有设置，可以考虑在这里让用户手动输入
    # HF_AUTH_TOKEN = input("请输入你的 Hugging Face 访问令牌: ")

# 模型 ID
model_id = "pyannote/speaker-diarization-3.1"
# 本地保存模型的目录
local_dir = "M:/meeting/speaker-diarization-3.1_downloaded_by_hf_hub" # 使用一个新目录，以免与git clone的混淆

print(f"准备从 Hugging Face Hub 下载模型 '{model_id}' 到本地目录: {local_dir}")
print(f"请确保您的 Hugging Face 令牌已设置，并且您已接受了模型的用户协议。")

try:
    # 列出仓库中的所有文件
    # 这一步需要网络连接和认证
    files_to_download = list_repo_files(repo_id=model_id, token=HF_AUTH_TOKEN)
    print(f"找到 {len(files_to_download)} 个文件，即将开始下载...")

    # 遍历并下载所有文件
    for file in files_to_download:
        print(f"  正在下载: {file}...")
        # hf_hub_download 会自动处理 LFS 文件
        hf_hub_download(
            repo_id=model_id,
            filename=file,
            cache_dir=local_dir, # 指定保存到这个目录，而不是默认缓存路径
            local_dir_use_symlinks=False, # 不使用软链接，直接复制文件
            token=HF_AUTH_TOKEN,
            # resume_download=True # 如果下载中断，可以尝试启用此选项
        )
        print(f"  完成下载: {file}")

    print(f"\n🎉 恭喜！模型 '{model_id}' 的所有文件已成功下载到 {local_dir}。")
    print("现在，您可以在 pyannote.audio 代码中使用这个新路径加载模型了。")

except Exception as e:
    print(f"\n❌ 错误：通过 huggingface_hub 下载模型失败！")
    print(f"错误信息: {e}")
    print("请确认以下事项：")
    print("1. 您的网络连接是否稳定，且代理（Clash for Windows）是否正常工作。")
    print("2. 您的 Hugging Face 访问令牌是否正确且具有读取权限。")
    print("3. 您是否已在 Hugging Face 网站上接受了 'pyannote/speaker-diarization-3.1' 和 'pyannote/segmentation-3.0' 的用户协议。")