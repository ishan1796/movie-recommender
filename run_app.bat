@echo off
title AI Movie Recommendation System
echo ========================================================
echo         AI MOVIE RECOMMENDATION SYSTEM
echo   Hybrid TF-IDF + Matrix Factorization Engine
echo ========================================================
echo.

set PYTHON_EXEC="C:\Users\ishan singh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if exist %PYTHON_EXEC% (
    echo Starting Streamlit Application using Python 3.12...
    %PYTHON_EXEC% -m streamlit run app.py
) else (
    echo Launching with default Python...
    streamlit run app.py
)

pause
