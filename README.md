# fliftbot_map_build

Переносимый пайплайн для преобразования записи M360, построения карты
standalone Cartographer и подготовки web/deploy-карты.

## Структура

```text
fliftbot_map_build/
├── src/map_build/       Python-код пайплайна
├── scripts/             вспомогательные shell-скрипты
├── tests/               unit-тесты
├── src/                 исходники, инструменты и standalone runner
└── configs/             конфигурации камеры, LiDAR и робота
```

Скомпилированные файлы, `__pycache__` и локальные пути не являются частью
исходников. Cartographer не поставляется в `vendor`: runner использует
системную библиотеку Cartographer.

## Требования

Поддерживаемое окружение — Linux с ROS 2 Humble. Нужны:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libeigen3-dev libcairo2-dev libzip-dev python3-venv python3-pip
```

Должен быть установлен Cartographer с CMake-конфигурацией, доступной после:

```bash
source /opt/ros/humble/setup.bash
find /opt/ros/humble -iname '*cartographer*Config.cmake' -o -name 'cartographerConfig.cmake'
```

Создание Python-окружения:

```bash
cd fliftbot_map_build
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Для обработки видео и ceiling-маркеров установите расширенный набор:

```bash
python -m pip install -r requirements-video.txt
```

Lidar-only запуск этих тяжёлых зависимостей не требует. Файл маркерной разметки
`labels2coords` является необязательным входом конкретного склада и передаётся
через `--labels2coords`. Без него observations сохраняются по tracker ID с
`code=-1`; геометрическая локализация и оценка высоты по трекам остаются доступны.
Для создания новых observations нужен также доступный модуль `CameraModel` из
`fliftbot_navigation`; существующие observations с сохранёнными лучами можно
обрабатывать без него.

## Сборка standalone Cartographer

После клонирования пересоберите runner на текущей машине:

```bash
cd fliftbot_map_build
source /opt/ros/humble/setup.bash
cmake -S src/standalone_cartographer \
      -B src/standalone_cartographer/build \
      -DCMAKE_PREFIX_PATH=/opt/ros/humble -DCMAKE_BUILD_TYPE=Release
cmake --build src/standalone_cartographer/build -j"$(nproc)"
```

Это наша обвязка над Cartographer, а не изменённая системная библиотека.
Изменённый runner находится в `src/standalone_cartographer/`, Lua-конфигурации — в `configs/cartographer/`.

## Полный запуск из сырых данных

```bash
cd fliftbot_map_build
source .venv/bin/activate
source /opt/ros/humble/setup.bash
python src/map_build/build_from_raw.py \
  --scan ../examples/scan.m360 \
  --odometry ../examples/fliftbot_move_logger_2026-10-07_10-10-04_Oct-07.log \
  --lidar-config configs/current_version/lidar.config \
  --canonical-output ../ros-data/recordings/my_run/canonical \
  --output ../ros-data/results --name my_run
```

Скрипт создаёт canonical recording, запускает Cartographer, строит web-карту,
валидирует bundle и при необходимости создаёт deploy-каталог.

## Повторный запуск из canonical recording

```bash
python src/map_build/build_from_canonical.py \
  --recording ../ros-data/recordings/my_run/canonical \
  --output ../ros-data/results --name my_run
```

Конфиг LiDAR обязателен. Без конфигов камеры и робота используются
`camera=(1.077, -0.274, 1.248) м`, `wheelbase=1.39 м`, поднятые вилы и
intrinsics из `configs/camera_intrinsics.json`. Основные параметры:
`--camera-x/y/z`, `--camera-roll/pitch/yaw`, `--lidar-x/y/z`,
`--lidar-roll/pitch/yaw`, `--wheelbase`, `--encoder-speed-sign -1|1`,
`--steering-sign -1|1`, `--mapping-yaw DEG`, `--odometry-time-offset SEC`, `--lidar-only`,
`--deploy-output DIR` и `--dry-run`.

Явные CLI-параметры всегда имеют приоритет над component config. Legacy-файл
`camera.txt` подключается через `--camera-position-file`: при наличии
`camera.config` координаты берутся из config, а roll/pitch — из `camera.txt`;
после этого индивидуальные `--camera-*` могут переопределить любое значение.

Отдельный повторный расчёт меток для готового запуска автоматически использует
конфиги и траекторию этого запуска; любой путь можно заменить явным аргументом:

```bash
python src/map_build/process_markers.py --run-dir ../ros-data/results/<run_id>
```

В каталоге `markers/` сохраняется собственный `run_parameters.json`.

Для независимой оценки высоты по трекам, проверки по участкам пути,
воспроизводимых команд и сравнения двух записей см.
[`docs/marker_height_block_cv.md`](docs/marker_height_block_cv.md).

Для текущего M360-тракта проверен контракт `mapping_yaw=180`,
`steering_sign=+1`; для move logger `.log` обычно нужен
`encoder_speed_sign=-1`, для encoder `.csv` — `+1`. Автоматический выбор знака
выполняется по метаданным canonical recording, явный параметр имеет приоритет.

## Отдельные инструменты и тесты

```bash
python src/map_build/convert_recording.py --help
python src/map_build/build_map.py --help
python src/map_build/build_web_map.py --help
python src/map_build/edit_web_map.py --help

python -m unittest discover -s tests -p 'test_*.py'
```

Ручное редактирование web-карты:

```bash
python src/map_build/edit_web_map.py \
  --map-dir ../ros-data/results/<run_id>/map \
  --output ../ros-data/results/<run_id>/web_map \
  --web-resolution-m-per-px 0.01 --gui
```

Подробное описание форматов и контрактов находится в `PIPELINE_TASKS.md`.

Положение вил задаётся параметром `--fork-position raised|lowered`. По
умолчанию используются поднятые вилы (`raised`). Для опущенных вил (`lowered`)
пайплайн добавляет 0.05 м к продольной координате камеры, продольной координате
лидара и wheelbase. Величину поправки можно переопределить параметром
`--lowered-forks-offset-m`.

`scripts/run_camera_calibration_experiments.sh` воспроизводит отдельную серию
калибровочных экспериментов. Перед запуском ему нужно передать пути через
переменные окружения, перечисленные в сообщениях самого скрипта (`NEW_MAP`,
`OLD_OBS_SOURCE`, `NEW_OBS`, `NEW_CAMERA`, `CAMERA_JSON`, `LIDAR_CONFIG`,
`VIDEO`, `TIMESTAMPS`, `LABELS`).
