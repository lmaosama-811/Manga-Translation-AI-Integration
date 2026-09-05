#!/usr/bin/env python3
"""
run_tunnel.py — Tự động chạy Cloudflare Tunnel và cập nhật GitHub Gist cho iPhone trên Linux / macOS / Windows.

Cách dùng:
    python run_tunnel.py
"""

import os
import re
import sys
import time
import shutil
import urllib.request
import urllib.parse
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
ENV_PATH = PROJECT_ROOT / ".env"

def load_env(env_path):
    env_vars = {}
    if not env_path.exists():
        print(f"❌ Không tìm thấy file .env tại {env_path}")
        sys.exit(1)
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip().strip('"').strip("'")
    return env_vars

def get_cloudflared_binary():
    # 1. Kiểm tra trong PATH
    if shutil.which("cloudflared"):
        return "cloudflared"
    
    # 2. Kiểm tra trong thư mục dự án
    local_bin = PROJECT_ROOT / ("cloudflared.exe" if sys.platform == "win32" else "cloudflared")
    if local_bin.exists():
        return str(local_bin)
    
    # 3. Tải về tự động nếu chưa có
    print("⏳ Đang tải cloudflared binary về thư mục dự án...")
    if sys.platform == "win32":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    elif sys.platform == "darwin":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
    else:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    
    try:
        urllib.request.urlretrieve(url, local_bin)
        if sys.platform != "win32":
            os.chmod(local_bin, 0o755)
        print(f"✅ Đã tải xong cloudflared: {local_bin}")
        return str(local_bin)
    except Exception as e:
        print(f"❌ Lỗi tải cloudflared: {e}")
        sys.exit(1)

def update_gist(gist_id, token, tunnel_url):
    print(f"⏳ Đang cập nhật URL mới lên GitHub Gist ({gist_id})...")
    api_url = f"https://api.github.com/gists/{gist_id}"
    payload = {
        "files": {
            "backend_url.txt": {
                "content": tunnel_url
            }
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "Manga-Translation-Tunnel"
        },
        method="PATCH"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status == 200:
                print("✅ Cập nhật GitHub Gist THÀNH CÔNG! iPhone sẽ tự động nhận URL mới.")
            else:
                print(f"⚠️ GitHub API trả về mã: {resp.status}")
    except Exception as e:
        print(f"❌ Không thể cập nhật Gist: {e}")

def main():
    print("=" * 60)
    print(" 🚀 Manga Translation — Cloudflare Tunnel Manager ")
    print("=" * 60)
    
    env_vars = load_env(ENV_PATH)
    gh_token = env_vars.get("GITHUB_TOKEN")
    gh_gist_id = env_vars.get("GITHUB_GIST_ID")
    
    if not gh_token or not gh_gist_id:
        print("⚠️ Thiếu GITHUB_TOKEN hoặc GITHUB_GIST_ID trong .env")
        sys.exit(1)
        
    cf_bin = get_cloudflared_binary()
    
    print("\n⏳ Đang khởi tạo Cloudflare Quick Tunnel hướng tới http://localhost:8000...")
    
    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", "http://localhost:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
    
    start_time = time.time()
    while time.time() - start_time < 45:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        # In ra log cloudflared nếu cần thiết
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break
            
    if not tunnel_url:
        print("❌ Không thể lấy URL Cloudflare Tunnel sau 45s. Vui lòng kiểm tra kết nối mạng.")
        proc.kill()
        sys.exit(1)
        
    print(f"\n🌐 URL Tunnel mới của bạn: {tunnel_url}")
    update_gist(gh_gist_id, gh_token, tunnel_url)
    
    print("\n" + "=" * 60)
    print(" 🎉 HỆ THỐNG ĐÃ SẴN SÀNG CHỜ IPHONE KẾT NỐI!")
    print(f" 📲 URL cho iPhone: {tunnel_url}")
    print(" 💡 Nhấn Ctrl + C để dừng Tunnel.")
    print("=" * 60 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Đang đóng Cloudflare Tunnel...")
        proc.terminate()
        proc.wait()
        print("Bảo trì tunnel kết thúc.")

if __name__ == "__main__":
    main()
