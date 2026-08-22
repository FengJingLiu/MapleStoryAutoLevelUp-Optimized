pyinstaller --noconsole --onefile src/main.py -p . ^
  --icon=media/icon.ico -n MapleStoryAutoLevelUp ^
  --hidden-import=pkg_resources.py2_warn ^
  --hidden-import=pkg_resources.extern ^
  --collect-submodules serial ^
  --additional-hooks-dir=pyinstaller_hooks ^
  --collect-all ultralytics ^
  --add-data "models\yolo\yolov8n_1024_rect_hero_mob_level_ge10_all_pets_2860_best.pt;models\yolo"
