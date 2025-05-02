@echo off
set count= 1
:loop
if %count% leq 10 (
    echo Starting testing script %count%/10...
    python play.py >> result.txt
    echo Finished testing script %count%/10
    set /a count+=1
    goto loop
)

echo All test scripts have been executed.