@echo off
title OPTIONS PAPER $5K (8054) - auto-restart
cd /d "%~dp0"
:loop
echo [%date% %time%] starting bot >> restarts_log.txt
python run_options.py
echo(
echo  *** BOT EXITED (crash or Ctrl+C). Auto-restarting in 10 seconds...
echo  *** To actually STOP this bot: close this window.
echo [%date% %time%] bot exited - auto-restarting >> restarts_log.txt
timeout /t 10 /nobreak >nul
goto loop
