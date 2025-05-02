@echo off
set count= 1
:loop
if %count% leq 10 (
    echo Starting training script %count%/10...
    python play.py >> result.txt
    echo Finished training script %count%/10
    set /a count+=1
    goto loop
)

echo All training scripts have been executed.

@REM 訓練完後即關機
@REM shutdown /s /f /t 150
@REM pause