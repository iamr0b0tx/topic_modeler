title semantic_segmentation - CICD
cls
@ECHO OFF

REM setting default
SET commit_msg="sync code with repo"

REM arg not empty set as value
IF [%1] NEQ [] SET commit_msg=%1

ECHO ----------getting requirements--------------------
pip freeze > requirements.txt

echo ----------------adding to git----------------------
git add .

echo --------------commiting to git --------------------
git commit -m %commit_msg%

echo -----------------git push--------------------------
git push

python main.py