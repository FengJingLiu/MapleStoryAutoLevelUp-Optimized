# 1024 YOLO monster model

`mob_1024_best.pt` is the final 50-epoch YOLOv8n checkpoint trained with
`imgsz=1024` and rectangular batches. Runtime inference selects only class 3
(`mob`).

- Source run: `yolov8n_1024_rect_batch12_v2`
- Ultralytics: `8.3.146`
- SHA-256: `0242F3BC9DC2BC8704AD6C88807C19373A176323D7CEB7FA5D8E65BE70976390`
- Classes: `character`, `environment`, `item`, `mob`, `npc`, `ui`
