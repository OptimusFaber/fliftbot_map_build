# fliftbot_map_build

Офлайн-пайплайн: запись **M360RAW1 / RPLRAW1 + одометрия → canonical →
Cartographer → карта и траектория → потолочные метки → web/deploy bundle**.
ROS bag не требуется. Рабочая интегрированная ветка — `main`.

## Установка

Поддерживаемая среда: Linux, Python **3.10+**, Cartographer (окружение ROS 2
Humble). Python-скрипты работают без запущенного ROS. Исходники Cartographer
в репозиторий не включены; нужна установленная системная библиотека с
`cartographerConfig.cmake`.

```bash
git clone git@github.com:OptimusFaber/fliftbot_map_build.git
cd fliftbot_map_build
sudo apt update
sudo apt install -y build-essential cmake pkg-config libeigen3-dev libcairo2-dev libzip-dev python3-venv python3-pip ros-humble-cartographer
source /opt/ros/humble/setup.bash
find /opt/ros/humble -iname '*cartographer*config.cmake'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cmake -S src/standalone_cartographer -B src/standalone_cartographer/build \
  -DCMAKE_PREFIX_PATH=/opt/ros/humble -DCMAKE_BUILD_TYPE=Release
cmake --build src/standalone_cartographer/build -j"$(nproc)"
```

Если CMake не находит Cartographer, установите его development-пакет либо
соберите библиотеку и добавьте её installation prefix в `CMAKE_PREFIX_PATH`.
Одного Python-пакета или бинарного ROS-узла недостаточно.

Для **нового распознавания видео** дополнительно:

```bash
python -m pip install -r requirements-video.txt
```

Модель `src/auto_map/model.onnx` и реализация `CameraModel` уже включены.
Пересчёт готовых `observations.json` выполняется с основными зависимостями,
без YOLO/Torch. Для GUI нужен графический сеанс; остальные этапы headless.
Все команды ниже запускаются **из корня репозитория** после активации `.venv`.
Пути `/data/...` замените своими.

## Два профиля построения меток

| Параметр | `warehouse` — высокие склады | `test_site` — низкие полигоны |
|---|---:|---:|
| Конфиг | `configs/markers/warehouse.json` | `configs/markers/test_site.json` |
| Минимальное перемещение камеры | 0,75 м | 0,25 м |
| Размер пространственного участка для block CV | 0,20 м | 0,05 м |
| Минимум различных положений камеры | 20 | 8 |
| Радиус группы положений камеры | 0,02 м | 0,02 м |
| Допустимая доля кадров с одного положения | 0,5 | 0,5 |
| Неопределённость / нестабильность высоты | 0,25 / 0,30 м | 0,25 / 0,30 м |
| Объединение повторных треков одной метки | 0,60 м | 0,60 м |

Для block CV требуется минимум шесть пространственных участков трека.
Оба профиля оценивают высоту отдельно по трекам, проверяют её методом
`block_cv` и исключают наблюдения с недостаточным движением из объединённой
оценки высоты. Профиль по умолчанию — **warehouse**. Один и тот же
`--marker-profile warehouse|test_site` поддерживают `build_from_raw.py`,
`build_from_canonical.py`, `build_map.py`, `run_pipeline.py`, `process_markers.py`.
Явные CLI-параметры имеют приоритет над профилем.

При неизменных угле обзора и скорости низкая метка обычно видна на более
коротком участке пути: для направленной вверх камеры размер видимой области
пропорционален разности высот метки и камеры. Это не универсальное ограничение
по времени: влияют скорость, ракурс, перекрытия и детектор. Поэтому профиль
полигона уменьшает требуемое **перемещение**, сохраняя проверку устойчивости.
Он проверен на синтетических коротких треках; точность на реальном низком
полигоне ещё нужно измерить. Складской профиль воспроизводит сохранённую
проверенную карту. Если реальные метки расположены ближе 0,60 м, уменьшите
`--merge-radius-m`, чтобы не объединять соседей.

Высота задаётся в метрах **от пола**, а не от камеры. В профилях нет
принудительной общей высоты и ограничения по высоте конкретного склада.
Можно добавить измеренный верхний предел `--max-marker-height-m 14.5`.
Для меток на одной измеренной плоскости `--ceiling-z 3.2` переключает расчёт
на заданную плоскость. Не сочетайте его с `--per-track-height`.
Для воспроизведения старого способа проверки доступны
`--height-quality-method legacy`, `--no-stationary-height-filter` и
`--no-per-track-height` (общая плоскость оценивается по трекам).

Слабые кандидаты могут сохраняться с `ceiling_z_m: null`: это отсутствие
надёжной высоты. Неоднозначные слабые треки могут остаться отдельными
кандидатами одной физической метки. `marker_id` назначается сортировкой и
не является постоянным ID между запусками. Декодерный код не используется
для геометрического объединения. Подробнее: [проверка высоты](docs/marker_height_block_cv.md).

## Геометрия оборудования

Профиль меток выбирается независимо от калибровки датчиков:

- `configs/bs_dephan/` — камера, DePHAN M360, погрузчик;
- `configs/bs_rplidar/` — камера, RPLIDAR, погрузчик;
- `configs/camera_intrinsics.json` — intrinsics, путь к ним указан в camera.config;
- `configs/cartographer/fliftbot_2d_offline.lua` — основной офлайн Cartographer;
  `fliftbot_2d.lua` — его базовые настройки.

Каждый комплект содержит `camera.config`, `lidar.config`, `liftbot.config`
с JSON-схемой `fliftbot.component_config.v1`. Координаты заданы относительно
оси вращения робота в метрах, ориентации — в градусах. Физическая ориентация
LiDAR и `mapping_orientation_deg` разделены. Проверьте mount и intrinsics
своего робота перед запуском: тип помещения не определяет калибровку.

В canonical обязательно нужен `lidar.config`. Камера и робот необязательны:
fallback камеры `(1.077, -0.274, 1.248) м`, yaw `-90°`, wheelbase `1.39 м`.
Для точных меток передавайте измеренную геометрию явно.

`--fork-position raised|lowered` по умолчанию `raised`. Для `lowered`
прибавляется `--lowered-forks-offset-m` (по умолчанию **0.05 м**) к camera.x,
lidar.x и wheelbase. Если config уже содержит геометрию опущенных вил,
не добавляйте эту поправку повторно. `--camera-x/y/z`, `--camera-roll/pitch/yaw`,
`--lidar-x/y/z`, `--lidar-roll/pitch/yaw`, `--wheelbase`, `--mapping-yaw`
переопределяют config. В 2D ненулевые LiDAR roll/pitch не поддерживаются.

Legacy-sidecar `--camera-position-file /data/camera.txt` содержит координаты
**в сантиметрах** на первой строке, roll/pitch[/yaw] **в радианах** на второй.
При наличии camera.config координаты берутся из config, roll/pitch — из
sidecar; затем применяются отдельные `--camera-*`.

## Полный запуск из сырых данных

```bash
python src/map_build/build_from_raw.py \
  --scan /data/scan.m360 --odometry /data/move_logger.log \
  --camera-config configs/bs_dephan/camera.config \
  --lidar-config configs/bs_dephan/lidar.config \
  --robot-config configs/bs_dephan/liftbot.config \
  --canonical-output /data/recordings/run01/canonical \
  --output /data/results --name run01 \
  --video /data/camera.mp4 --frame-timestamps /data/frame_timestamps.csv \
  --marker-profile warehouse
```

Для RPLIDAR замените scan на `/data/scan.rplidar`, одометрию на
`/data/wheel_encoder_data.csv`, комплект на `configs/bs_rplidar/`.
Формат LiDAR определяется по заголовку файла, не по расширению.
Без `--video/--frame-timestamps` или `--observations` этап меток пропускается.
`--lidar-only` отключает одометрию в Cartographer; файл encoder всё равно
нужен для canonical-контракта. Сырой input и ранее созданные результаты
не перезаписываются; для новой конвертации выбирайте новый canonical-каталог.

Вместо видео можно передать `--observations /data/observations.json`.
`--labels2coords /data/labels2coords.json` необязателен и используется только
для декодирования видео. Без него сохраняются track ID и `code=-1`.
GT-координаты не участвуют в построении карты или подборе геометрии.

Время камеры и LiDAR должно находиться в одной Unix-шкале. CSV кадров:
`frame_index,time_take_img_sec`; observations могут хранить `frame_timestamps`
внутри JSON, иначе рядом нужен `frame_timestamps.csv`. Encoder CSV:
`timestamp_sec,linear_velocity_m_per_s,wheel_angle_rad`. Move logger `.log`:
`enc_t,enc_v,enc_dgr`, время `YYYY_MM_DD_HH_MM_SS_microseconds` в Europe/Moscow,
угол в градусах. Поле `enc_t` со значением 1970 игнорируется.

Конвертация M360 по умолчанию `--m360-point-time-mode packet-host`;
экспериментальная временная модель — `device-linear`. RPLIDAR по умолчанию
`--rplidar-point-time-mode azimuth-device-end` восстанавливает время лучей
по азимуту. Другие режимы перечислены в `--help`. Интервал
`--lidar-start-unix/--lidar-end-unix` поддержан для M360; для RPLRAW1
обрезайте готовую canonical-запись инструментом `scripts/slice_canonical_interval.py`.

## Повторный запуск из canonical

```bash
python src/map_build/build_from_canonical.py \
  --recording /data/recordings/run01/canonical \
  --output /data/results --name run01 \
  --observations /data/observations.json --marker-profile test_site
```

Конфиги по умолчанию берутся из canonical; любой можно заменить явным
`--camera-config`, `--lidar-config`, `--robot-config`. Canonical содержит
`horizon/*.npz`, `encoder.jsonl`, конфиги и `conversion_report.json`.
NPZ содержит `xy` (N×2, метры), `t` (времена лучей), `stamp` (время скана).
Имена пакетов сохраняют временной порядок.

`--encoder-speed-sign -1|1` и `--steering-sign -1|1` задают соглашения знаков.
Автовыбор скорости: metadata `odometry_kind=log` → `-1`, `csv` → `+1`, иначе
значение lidar.config. Для M360 проверен `mapping_yaw=180`, steering sign `+1`;
для RPLIDAR mount yaw обычно `0`. При необычном контракте записи используйте
явные параметры успешного запуска.

`--odometry-time-offset SEC` добавляется к времени энкодера;
`--marker-time-offset SEC` — к времени кадров, иначе берётся
`time_offset_s` камеры. Это разные поправки. Пропуски encoder 0,5…1,5 с
линейно интерполируются и отражаются в `map/run_stats.json`; пропуск свыше
1,5 с во время движения отклоняется валидатором. Интерполяция не заменяет
качественную запись движения.

`--dry-run` валидирует canonical и создаёт входы/параметры запуска без
Cartographer. В `build_from_raw.py --dry-run` сырые данные сначала
**конвертируются**. Запуски записываются в `OUTPUT/YYYYMMDD_HHMMSS_NAME`.

## Пересчёт только меток

```bash
python src/map_build/process_markers.py \
  --run-dir /data/results/20261030_120000_run01 \
  --observations /data/observations.json \
  --marker-profile warehouse
```

Траектория, карта, camera sidecars и lidar config берутся из запуска.
Если observations уже указаны в `run_parameters.json` или созданы в
`input/video/`, повторно задавать путь не нужно. Для другого источника можно
переопределить `--trajectory`, `--map-dir`, `--camera-position`, `--camera-json`,
`--lidar-config`; для отдельного результата используйте `--out-dir`.
Повторная обработка с новым профилем требует явного `--marker-profile`;
по умолчанию всегда выбирается `warehouse`.

`--time-offset-s` переопределяет сохранённую поправку камеры. `--camera-config`
перегенерирует sidecars и берёт поправку времени из него; сохранённые лучи
при этом **не перестраиваются**. При изменении калибровки заново создайте
observations или используйте `scripts/rebuild_observation_rays.py`.

Основные результаты в `markers/`:

- `marker_points_stored_ray_corrected.json` — XYZ, качество высоты, треки и голоса кодов;
- `stored_ray_corrected_map.png`, `stored_ray_corrected_map_closeup.png` — карты;
- `run_parameters.json` — итоговые параметры, команда, профиль, его содержимое и SHA-256;
- `markers_map.json`, `markers_map.csv`, `markers_map.png` — координаты в
  pbstream/local/web, если у запуска есть `web_map/`.

При пересчёте стандартного `RUN/markers/` экспорт координат, превью и
`artifacts.json` обновляются. При отдельном `--out-dir` создаётся только
маркерный результат, исходный bundle не меняется. Проверяйте поле `status`:
`abstained` означает, что карта меток не принята (меньше трёх кандидатов или
нет общей плоскости в соответствующем режиме); успешный код CLI сам по себе
не гарантирует принятую карту меток.

## Web-карта, origin и передача в навигацию

Полный запуск создаёт `map/` с PBStream, occupancy, траекторией и `track.png`;
`web_map/` с dark/light JPEG, metadata и матрицами; параметры, отчёты
валидации и `artifacts.json` с SHA-256. `map/processing.log` и glog-файлы
содержат диагностику Cartographer, `map/run_stats.json` — счётчики сканов,
одометрии и constraints. Прогресс выводится в терминал.

Для ручного crop/rotation:

```bash
python src/map_build/edit_web_map.py \
  --map-dir /data/results/RUN/map --output /data/results/RUN/web_map --gui
```

Для переноса origin при полном запуске добавьте
`--origin-marker-id marker_001` или `--origin-map-xy X Y`,
`--origin-yaw-deg DEG`. Отдельная команда:

```bash
python src/map_build/set_map_origin.py \
  --web-map /data/results/RUN/web_map --origin-map-xy 1.0 2.0 --yaw-deg 0
```

Меняется система координат, пиксели не перемещаются. После ручного изменения
web-карты/origin перезапустите `process_markers.py` для обновления координат
меток, затем валидацию и упаковку:

```bash
python src/map_build/validate_bundle.py /data/results/RUN --json /data/results/RUN/validation_report.json
python src/map_build/package_bundle.py --run /data/results/RUN --output /data/deploy/run01
```

Либо передайте `--deploy-output /data/deploy/run01` в полный запуск.
Deploy содержит `map.pbstream`, occupancy, web-изображения, матрицы,
`map_metadata.json`, `origin_reference.json` и manifest с SHA-256.
PBStream обязателен для continuous Cartographer localization; JPEG/PGM его
не заменяют. Копирование в `realtime_mapping` и переключение активной карты
выполняются отдельно. Подробности координат — [контракт](docs/pipeline_contract.md).

## Проверка и структура

```bash
python -m unittest discover -s tests -p 'test_*.py'
ctest --test-dir src/standalone_cartographer/build --output-on-failure
python src/map_build/validate_recording.py /data/recordings/run01/canonical
python src/map_build/validate_bundle.py /data/results/RUN
```

Все публичные инструменты имеют `--help`: `convert_recording.py`,
`build_from_raw.py`, `build_from_canonical.py`, `build_map.py`,
`build_web_map.py`, `edit_web_map.py`, `process_markers.py`, `set_map_origin.py`,
`validate_recording.py`, `validate_bundle.py`, `package_bundle.py`.
`scripts/` содержит отдельные инструменты анализа/конвертации с собственными CLI.

```text
src/map_build/                  этапы пайплайна и общие профили меток
src/standalone_cartographer/    C++ runner, использующий системный Cartographer
src/auto_map/                   YOLO, BoT-SORT, decoder и модель ONNX
src/fliftbot_navigation/        CameraModel
src/realtime_mapping_scripts/   конвертеры raw LiDAR
src/fliftbot_mapping/evaluation/ геометрия меток и отдельная оценка по GT
configs/                       оборудование и два профиля меток
scripts/                       вспомогательные инструменты
tests/                         регрессионные тесты
docs/                          контракты, метод высоты, результаты проверки
```

Старые датированные эксперименты, sweep-конфиги и временные диагностические
скрипты убраны из текущей версии; отслеживаемая история доступна через Git.
Видео, raw/canonical-записи, GT и результаты не включаются в исходники.
Сборка, `.venv`, cache и `results/` исключены через `.gitignore`.
Результаты проверки этой фиксации: [verification](docs/verification.md).
