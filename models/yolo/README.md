# 1024 YOLO models

`yolov8n_1024_rect_hero_mob_16000_v6_best.pt` is the default checkpoint for
both monster and Hero detection. Runtime inference is shared between class 0
(`mob`) and class 1 (`hero`); the Hero class uses its independent `0.85`
confidence threshold.

- Source run: `yolov8n_1024_rect_hero_mob_16000_v6_cleanlabels`
- Input size: `1024`
- Classes: `mob`, `hero`
- SHA-256: `2F6426A186059B782D841D79775AC6803F640BA83070D7A514613578E0DEEFD7`

## Crocodile Pond 2 fine-tune

`hero_mob_crocodile_pond_2_10000_best.pt` fine-tunes the two-class checkpoint
above on 10,000 synthetic Crocodile Pond 2 samples at `imgsz=1024` with
rectangular batches. The dataset contains 9,000 bottom-platform Hero samples,
500 random-platform samples, and 500 rope samples, plus unlabelled drops and
other players as hard negatives.

- Source run: `runs/yolov8n_1024_rect_crocodile_pond_2_10000`
- Effective fine-tuning epochs: 15
- Ultralytics: `8.3.146`
- Classes: `mob`, `hero`
- Validation (488 images): P `0.998`, R `0.994`, mAP50 `0.995`, mAP50-95 `0.988`
- SHA-256: `EE8C7D36C6B4FA7A0BBD1D233D5B2D830E814C462FC0E7B32E8B9E6B452D41BA`
