pyinstaller --noconsole --onefile src/main.py -p . ^
  --icon=media/icon.ico -n MapleStoryAutoLevelUp ^
  --hidden-import=pkg_resources.py2_warn ^
  --hidden-import=pkg_resources.extern ^
  --collect-all ultralytics ^
  --add-data "models\yolo\mob_1024_best.pt;models\yolo"
