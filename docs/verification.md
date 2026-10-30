# Проверка фиксации от 2026-10-30

Версия отмечена тегом `marker-map-2026-10-30`, интеграционная ветка — `main`.

## Автоматические проверки

- `python3 -m unittest discover -s tests -p 'test_*.py'`: **16 тестов, OK**.
  Проверены координатные матрицы и origin, CSV/JSON-экспорт неизвестной высоты,
  оба профиля меток на коротком треке, отказ от высоты при недостаточном
  числе положений камеры, encoder gaps, M360 timestamp, RPLIDAR beam timing,
  цепочка raw → canonical → materialized inputs и передача профиля/времени/вил.
  Повторная запись в заполненный canonical-каталог отклоняется.
- CMake configure и **полная пересборка с `--clean-first`**: успешно.
- CTest `standalone_help`: **1/1 passed** после пересборки.
- `--help` всех 12 CLI в `src/map_build/` и 18 вспомогательных скриптов:
  успешно. Для гибридных видео используется общий `process_markers.py`.
- JSON-конфиги читаются, относительные ссылки на intrinsics существуют;
  `git diff --check` проходит. Модель ONNX и таблицы декодера сохранены как
  необходимые runtime assets. Raw-записи и generated outputs в Git не добавлены.

## Точное воспроизведение складской карты

Источник — выбранный результат
`/home/optimus/recordings_2nd_try/result/20261029_124432_nav_video/marker_map_block_cv/`.
Вызван публичный `process_markers.py` с профилем `warehouse`, теми же
observations, camera sidecar, lidar config, траекторией, `time-offset-s=0` и
`max-marker-height-m=14.5`.

Результат принят: **411 меток, 947 принятых треков**. Сохранённые и новые
`markers`, `counts`, `track_height_quality`, `weak_height_associations` и
`quality_thresholds` совпадают полностью, включая XYZ и неизвестные высоты.
Это регрессия алгоритма, а не новая оценка точности по GT.

Команда воспроизведения при наличии исходной записи:

```bash
RUN=/home/optimus/recordings_2nd_try/result/20261029_124432_nav_video
OBS_ROOT=/home/optimus/recordings_2nd_try/marker_observations_new
python3 src/map_build/process_markers.py \
  --run-dir "$RUN" --marker-profile warehouse \
  --observations "$OBS_ROOT/observations_decoded.json" \
  --camera-position "$OBS_ROOT/camera_position.txt" \
  --lidar-config "$RUN/input/lidar.config" \
  --max-marker-height-m 14.5 --time-offset-s 0 \
  --out-dir /tmp/fliftbot-map-build-verification/warehouse
```

## Сквозной прогон на реальных LiDAR/encoder

Выбраны первые **3000 NPZ-пакетов** той же записи; encoder ограничен
соответствующим интервалом с запасом по секунде. Геометрия взята из
`run_parameters.json` исходного запуска, оригинальные входы не менялись.
`build_from_canonical.py` с системным Cartographer создал:

- карту PBStream/PGM/PNG/YAML и траектории;
- **2644** поданных encoder samples, **925** Cartographer constraints;
- web-карту и согласованные матрицы, bundle validation: **accepted**;
- deploy-каталог: все **11 файлов** manifest проверены по размеру и SHA-256.

Затем `process_markers.py` на **новой** траектории построил принятую карту
из **16 меток / 24 треков**, обновил JSON/CSV координат, превью и
`artifacts.json`. Все **38 файлов** итогового artifacts manifest проверены
по размеру и SHA-256. Входная запись принята с предупреждениями о небольших
временных gaps; максимальный encoder gap в этом фрагменте — 0,201307 с.
Этот короткий прогон проверяет работоспособность всей цепочки и не является
сравнением геометрической точности полной карты.

Дополнительно 12 реальных кадров обработаны YOLO/BoT-SORT без codebook:
сохранено **72 observations, все 72 с лучами**, CameraModel загрузился.
Полный многоминутный видеофайл заново не распознавался.

## Профиль низкого полигона

Синтетическая сцена содержит три метки на высоте 3 м, камеру на 1 м и
41 наблюдение на пути 0,40 м. `test_site` восстанавливает все три высоты
с погрешностью менее 1e-6 м; `warehouse` отклоняет эти короткие треки.
При заведомо недостаточном числе независимых положений камеры высоты
становятся `null`. Реальной записи низкого полигона в эту проверку не
включено: его полевую точность ещё следует проверить, особенно при близких
соседних метках и коротком времени видимости.

## Окружение и локальные артефакты

Проверка выполнена на Linux с Python 3.14, ROS 2 Humble Cartographer
2.0.9004, NumPy 2.5.2, SciPy 1.18.1, Pillow 12.3.0, OpenCV 5.0.0.93,
Ultralytics 8.4.127, Torch 2.13.0, ONNX Runtime GPU 1.29.0. Это описание
проверенной среды, не требование устанавливать именно GPU runtime.

Временные отчёты, изображения и smoke inputs находятся в
`/tmp/fliftbot-map-build-verification/`; их можно удалить после просмотра.
Исторические sweep-скрипты, старые калибровки и разовые диагностики сохранены
в локальном архиве
`/home/optimus/ros-projects/tools/fliftbot_map_build_archive/20261030/`
с `archive_manifest.json`. Файлы, ранее отслеживаемые Git, также доступны
в истории. Рабочие исходные записи и выбранные результаты не удалялись.
