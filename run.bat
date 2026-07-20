@echo off
:: Manga Translator - Start server + Cloudflare tunnel
:: Usage: .\run.bat             (full: server + tunnel + Gist)
::        .\run.bat -Local      (local only, no tunnel)
set "PYTHONUTF8=1"
pushd "%~dp0"
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0run_tunnel.ps1" %*
popd
