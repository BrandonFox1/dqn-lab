@echo off
REM One-click: install, test, train on the small maze, record a GIF.
setlocal
cd /d "%~dp0"
python -m pip install -r requirements.txt || goto :err
python -m pytest -q || goto :err
python -m pacdqn.train --maze small --ghosts 1 --steps 150000 --out runs\small || goto :err
python -m pacdqn.play --run runs\small --gif runs\small\play.gif --episodes 3 || goto :err
echo.
echo Done. Open runs\small\play.gif to watch the agent; runs\small\eval.csv has the learning curve.
pause
exit /b 0
:err
echo Something failed (see above).
pause
exit /b 1
