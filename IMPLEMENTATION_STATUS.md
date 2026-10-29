# Статус реализации

> Этот файл описывает ранний этап пайплайна. Текущая проверка высоты меток и
> её ограничения описаны в `docs/marker_height_block_cv.md`.

## Работает

- `convert_recording.py`: M360RAW1 → `horizon/*.npz`, move logger/encoder CSV →
  `encoder.jsonl`, синхронизация `.log` по ближайшим LiDAR timestamp, SHA-256,
  отчёт и самодостаточные копии конфигов/интринсиков камеры.
- `validate_recording.py`: порядок и покрытие времени, дубли, gaps, частоты,
  доля движения, UTC-представление границ.
- `build_from_raw.py`: полный вход из сырых данных.
- `build_from_canonical.py`: validation → Cartographer → web map → markers/origin
  → bundle validation → optional deploy catalog.
- `build_web_map.py`: корректное отделение unknown от occupied, фильтр далёких
  разреженных лучей для crop, выравнивание по доминирующим ортогональным
  стенам, 1 см/px по умолчанию, точные матрицы координат.
- `edit_web_map.py`: ручной crop/rotation, включая интерактивный выбор crop
  через `--gui`, с пересчётом матриц.
- `set_map_origin.py`: origin по координате или `marker_id` без перемещения
  пикселей.
- `package_bundle.py`: отдельный каталог карты и `origin_reference.json`; файлы
  `realtime_mapping` автоматически не изменяются.
- Автотест инвариантов матриц и origin.

## Ещё требует работы/полевой проверки

- Порог уверенности автоматического выравнивания стен не зафиксирован в ТЗ;
  сейчас сохраняются число линий и разброс углов, но решение не блокируется.
- GUI поддерживает визуальный crop; выбор origin удобнее пока делать отдельной
  командой `set_map_origin.py`.
- Новый majority-vote и экспорт marker coordinates реализованы, но полный
  прогон на видео с конфликтующими decoder labels ещё не выполнен.
- Выполнен полный lidar-only smoke run на `examples/scan.m360`: Cartographer,
  web/post-processing, bundle validation и deploy завершились успешно. Режим с
  одометрией и видео всё ещё следует проверить отдельными интеграционными
  прогонами перед использованием как production-процедуры.
