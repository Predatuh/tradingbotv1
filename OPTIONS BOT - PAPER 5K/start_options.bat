@echo off
title Options Bot (auto-restart)
echo Options bot with auto-restart. Close this window to stop for good.
:loop
python run_options.py
echo.
echo Bot stopped. Restarting in 30 seconds... Ctrl+C to stay stopped.
timeout /t 30
goto loop
