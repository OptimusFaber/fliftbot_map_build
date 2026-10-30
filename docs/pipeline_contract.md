# Контракт данных и координат

## Этапы

`build_from_raw.py` вызывает `convert_recording.py`, затем
`build_from_canonical.py`. Последний выполняет validation → `build_map.py`
(обёртка над `run_pipeline.py`) → web map → экспорт меток → необязательный
origin → bundle validation → manifest → необязательный deploy.
`process_markers.py` выполняет только маркерный этап по готовой траектории.

Raw scan выбирается по magic `M360RAW1` или `RPLRAW1`; содержимое одометрии
определяет `.log`/CSV. Canonical-каталог хранит пакеты `horizon/*.npz`,
`encoder.jsonl`, конфиги оборудования, intrinsics, `conversion_report.json`.
В encoder строки имеют `enc_unix` (Unix seconds), `enc_v` (м/с),
`enc_dgr` (градусы), `enc_available` (bool). LiDAR NPZ: `xy` N×2 в метрах,
`t` N времён лучей и `stamp` времени скана; дополнительные `xyz`, `intensity`,
`device_t` сохраняются при наличии. Порядок файлов должен совпадать со временем.
Для защиты от смешивания записей используйте отдельный canonical-каталог
для каждой конвертации.

Входы mapping materialize как обычные файлы/hardlinks (copy при другом
filesystem). Guard `check_standalone_inputs.py` допускает только canonical
LiDAR/encoder и lidar config; соседние GT/nav-артефакты не читаются runner.

## Системы координат

- `pbstream_map`: метры в исходной системе Cartographer, траектория LiDAR.
- `local_map`: метры относительно выбранного origin и направления осей.
- `web_raster`: пиксели JPEG, `u` направлена вправо, `v` вниз.

Матрицы 3×3 действуют на столбец `[x, y, 1]`:

```text
local_xy = pbstream_to_local_matrix @ pbstream_xy
raster_uv = map_transform_matrix @ local_xy
raster_uv = pbstream_to_raster_matrix @ pbstream_xy
pbstream_to_raster_matrix = map_transform_matrix @ pbstream_to_local_matrix
```

Матрицы сохраняются как `.npy` и внутри `web_map/map_metadata.json`.
При смене origin исходная картинка и `pbstream_to_raster_matrix` сохраняются,
меняются две другие матрицы. Траектория Cartographer имеет
`tracking_frame=lidar`, время `timestamp_universal` в 100 нс от 0001-01-01:
`unix_s = (timestamp_universal - 621355968000000000) / 1e7`.
`trajectory_robot_axis.json` переносит позы к оси вращения робота;
`trajectory_wheel_odometry.json` — отдельная интеграция encoder для сравнения.

## Метки

Observations содержат `data`: список кадров с объектами `track_id → {ray,
code, center, ...}`. `ray` — нормированный луч после CameraModel, поворот
камеры уже включён. Нельзя применять его второй раз. `frame_timestamps`
в JSON либо соседний `frame_timestamps.csv` задаёт Unix-время кадров.
Поправка `time_offset_s` **добавляется** к этому времени.

Смещение камеры относительно LiDAR поворачивается body yaw, полученным из
траектории LiDAR за вычетом физического mount yaw. Луч поворачивается тем же
body yaw и независимой `ray_correction_deg.yaw`. Высота камеры задана от пола.
Проверка физически допустимой высоты трека сейчас использует диапазон
`camera_z + 0.5 … camera_z + 15 м`; `--max-marker-height-m` может сузить его.

Новый профиль по умолчанию `warehouse` включает независимую высоту по трекам
и block CV. `test_site` допускает меньшую базу и короткие пространственные
участки. Объединение треков — complete link по XY: все центры группы должны
попадать в радиус. Слабые треки присоединяются к надёжной высоте только при
однозначной близкой проекции; иначе выдаётся кандидат с неизвестной высотой.
Физическая близость слабых треков не доказывает их тождественность.
Декодированный код — metadata, не основание слияния.

`marker_points_stored_ray_corrected.json` содержит геометрию и диагностику;
`markers_map.json`/CSV — её преобразование в три системы координат.
`null` высота переносится без подстановки общей высоты. `marker_id`
назначается заново сортировкой результата. Профиль, его SHA-256,
полное содержимое, применённые CLI overrides и фактическая команда хранятся
в `markers/run_parameters.json`. Корневой `run_parameters.json` фиксирует
геометрию и параметры полного запуска.

`abstained` в JSON означает отсутствие принятой карты меток. Код CLI сообщает
об успешном выполнении вычисления, поэтому потребитель должен отдельно
проверять `status` и наличие наблюдаемой высоты.

## Валидация и deploy

`validate_recording.py` проверяет поля/форму, порядок времени, временное
пересечение, gaps и движение. Пропуски encoder 0,5…1,5 с интерполируются
линейно; большие пропуски при движении отклоняются валидатором.
`validate_bundle.py` проверяет обязательные map/web-файлы, форму и конечность
матриц, обратимость основных преобразований, композицию и согласованность
metadata с изображением/матрицами. Он не оценивает геометрическую точность
склада по GT. Тесты отдельно проверяют round trip координат и перенос origin.
`artifacts.json` содержит размеры и SHA-256 артефактов; входные LiDAR-пакеты
в manifest не перечисляются.

Deploy — отдельный каталог с `map.pbstream`, `map.pgm/png/yaml`, web JPEG,
матрицами, metadata, `origin_reference.json` и manifest. Reference имеет
схему `fliftbot.map_origin_reference.v1`, frame `cartographer_map`, положение
origin и угол осей в исходной карте. Для continuous localization необходим
PBStream. Runtime-файлы исходных видео, raw scan, GT и experiments в deploy
не включаются. Упаковка не переключает карту и не меняет конфиги
`fliftbot_lidar_nav`.


Deploy включает `navigation_config.json` (`fliftbot.navigation_config.v1`):
LiDAR type/points frame, scan_to_base XY/yaw, максимальная дальность и
одометрию (speed sign, steering sign/zero, wheelbase, drive_wheel model).
Источником служит эффективный `run_parameters.json`, включая CLI overrides.
Неполный RUN не упаковывается, непустой deploy не перезаписывается.
`fliftbot_lidar_nav/nav.sh check-map --map-dir DEPLOY` проверяет SHA-256,
конфиги, occupancy и соответствие reference матрице.
Связанный pipeline: [simulation → map → localization](../../fliftbot_lidar_nav/docs/pipeline.md).
