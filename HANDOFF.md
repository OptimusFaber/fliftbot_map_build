# Handoff: mapping pipeline (2026-10-16)

> Исторический срез на 2026-10-16. Текущие команды находятся в `README.md`,
> а актуальная проверка высоты меток — в `docs/marker_height_block_cv.md`.

Короткий контекст для продолжения работы другим агентом. Основной проект:
`/home/optimus/ros-projects/fliftbot_map_build`.

## Что собрано

- Единый pipeline: raw `scan.* + .log/.csv` → canonical `horizon/*.npz + encoder.jsonl` → Cartographer → `map.pbstream/pgm/png/yaml`, `trajectory.json`, track image, web-map и параметры запуска.
- Entry points: `src/map_build/build_from_raw.py`, `build_from_canonical.py`, `build_map.py`, `build_web_map.py`, `edit_web_map.py`.
- Конфиги камеры/LiDAR/погрузчика приведены к JSON-схеме `fliftbot.component_config.v1`.
- Raw `.log` режется по ближайшим Unix-временам начала/конца LiDAR; `.csv` можно брать полностью.
- Есть GUI web-map editor: ручной контур, origin, приблизительные Ox/Oy с ортогонализацией, сохранение за один запуск.
- Поддержаны гибридные записи и несколько видео для маркеров.
- Добавлен `--fork-position raised|lowered` (default `raised`) и `--lowered-forks-offset-m` (default `0.03`). Для `lowered` offset добавляется к camera.x, lidar.x и wheelbase.
- Скрипты экспериментов: `scripts/run_tail_experiments*.sh`, `run_lowered_forks_sweep.sh`, `run_camera_calibration_experiments.sh`.

## Проверенный BS mapping contract

Конфиги брать из `configs/bs_version/`. Для последних BS-заездов рабочая комбинация:

```text
LiDAR mapping yaw: 180 deg
steering_sign:     +1
forks:             lowered
base wheelbase:    1.385 m
fork offset:       0.030 m
effective WB:      1.415 m
camera x:          1.107 m
lidar x:           1.083 m
max range:         18 m (для выбранного tail-эксперимента)
```

Важно: знак скорости нельзя угадывать только по расширению. Исторически `.log/.csv` имели разные соглашения; всегда смотреть `run_parameters.json` конкретного успешного запуска. У выбранного запуска `encoder_speed_sign=+1`, `steering_sign=+1`, odometry включена.

Лучший актуальный 10-min run:

```text
/home/optimus/ros-projects/ros-data/results/real/20261015/tail_10min_18m_lowered_forks/
20261015_004629_tail_10min_18m_forks_down_3cm_wb1385
```

Его `run_parameters.json` — источник истины по геометрии. Аналогичный 12-min run:

```text
/home/optimus/ros-projects/ros-data/results/real/20261015/tail_12min_18m_lowered_forks/maps/
20261015_004734_tail_12min_18m_forks_down_3cm_wb1385
```

## Что выяснено про карты

- Длинные 30-min BS-записи дают дуги/волны вдоль одной стены даже в lidar-only; это не только одометрия.
- Ограничение дальности до 16–20 m улучшает карту; 18 m выбран как практический вариант.
- Скорость сама по себе не объясняет плохую карту: быстрые заезды `20261007/115022_crocus` и `142322_crocus` строились хорошо.
- Tail 6/8/10 min и гибриды обычно лучше полного длинного заезда. Для текущей работы выбран tail 10 min.
- Сравнение encoder/cartographer показало, что одометрия действительно поступает; проблемы карты не сводятся к «Cartographer не ест encoder».

### Экспериментальное время точек M360 (2026-10-18)

По `instructions.pdf`, UDP `TimeStamp` относится к первой точке пакета, а
`t(x) = TimeStamp + x / (HZ * 2300)`. Добавлен opt-in режим
`--m360-point-time-mode device-linear`; default `packet-host` оставлен без
изменений. Режим также исправляет декодирование little-endian unsigned 32.32
TimeStamp (младшее 32-битное слово — доли секунды, старшее — секунды), робастно
привязывает device clock к Unix host receive clock и сохраняет индивидуальное
время каждой точки. Для коротких экспериментов доступны
`--lidar-start-unix/--lidar-end-unix`. Проверка на BS-записи: 2300 точек дают
2300 уникальных времён, медианный шаг при 25 Hz около 17.4 мкс; старые карты
этим изменением автоматически не затрагиваются.

## Маркеры: текущая реализация

Основной video exporter:
`src/auto_map/get_video_data.py` (YOLO + BoT-SORT). Detector-only exporter существует, но не даёт постоянных track ID.

Последнее исправление:

1. Все YOLO-боксы поступают в BoT-SORT, чтобы край кадра не обрывал track ID.
2. Перед декодированием/экспортом луча отбрасываются боксы, касающиеся края, с aspect ratio `<0.72` и адаптивные small-area outliers.
3. В `ceiling_marker_localization_stored_ray_corrected.py` robust height consensus теперь реально применяется и в `--per-track-height`; раньше маска считалась, но игнорировалась.

Актуальные observations с `camera_backup.json`, опущенными вилами и исправленной фильтрацией:

```text
/home/optimus/ros-projects/ros-data/recordings/real/20261013/141857_dephan_m360_bs/
observations_camera_backup_lowered/observations.json
```

Маркерная карта выбранного 10-min run находится в `markers/`. Старые контрольные варианты сохранены рядом:

- `markers_before_detection_filter/` — старые observations;
- `markers_filter_before_tracker/` — ошибочный фильтр до BoT-SORT;
- `markers_tracker_fixed_before_height_gate/` — исправленный tracker, но без height outlier gate.

Marker IDs назначаются сортировкой результата и **нестабильны между перезапусками**; одинаковый номер не гарантирует одинаковую физическую метку.

## Известная физическая истина и текущая ошибка

На складе 43 маркера. Реальные высоты: `7.149…7.262 m`.

Для физического четырёхугольника:

```text
001–002 = 3.02 m
002–005 = 4.89 m
005–003 = 3.04 m
003–001 = 4.89 m
```

Текущий исправленный 10-min результат:

```text
001–002 = 3.063 m  (+0.043)
002–005 = 4.693 m  (-0.197)
005–003 = 3.157 m  (+0.117)
003–001 = 4.571 m  (-0.319)
```

Ошибки имеют разные знаки, поэтому единый SE(2), scale или высота их не исправляют. Диагностический прогон с общей высотой `7.205 m` дал `3.098 / 4.616 / 3.044 / 4.638 m`: одна сторона исправилась, остальные нет. Главный нерешённый вопрос — калибровка лучей/времени/траектории.

Следующий корректный шаг без подгона по известным расстояниям:

1. Оценить video↔trajectory time offset по минимальной многовидовой несогласованности одного track ID.
2. Проверить intrinsics/distortion `camera_backup.json` на фактическом разрешении видео.
3. Независимо откалибровать camera↔LiDAR 6DoF для опущенных вил.
4. Оптимизировать time offset + roll/pitch/yaw + lever arm только по схождению лучей и физическому диапазону высот; реальные четыре расстояния оставить hold-out validation.
5. После этого разбирать missing/duplicate markers; не подбирать фильтр искусственно под число 43.

## Артефакт четырёх маркеров

Каталог:

```text
<10-min-run>/markers_selected_001_002_003_005/
```

Содержит карту только с четырьмя расстояниями, `distances.json` и `observations_top1.json`. Последний хранит по одному лучшему кадру на маркер (largest valid bbox) плюс 0/1-based frame number, Unix timestamp, время внутри полного видео и source track ID. Генератор: `scripts/export_marker_subset.py`.

Top-1 кадры полного видео:

```text
001: frame 14583 (0-based), 00:30:21.269
002: frame 14532,           00:30:14.850
003: frame 14421,           00:30:00.930
005: frame 10638,           00:22:13.730
```

## Осторожно

- Не сравнивать marker IDs между каталогами без spatial/track matching.
- Не применять camera mount rotation второй раз: stored ray уже содержит CameraModel rotation.
- `trajectory.json` — pose LiDAR tracking frame; lever arm переводится через body heading с учётом LiDAR mount yaw.
- Старый хороший результат `170557_wheelbase_yaw180_ab` полезен как reference, но построен по другому/исторически некорректному odometry contract и не должен быть GT для настройки нового pipeline.
- В workspace нет полезной git-истории (много untracked файлов); перед крупными изменениями сохранять результаты отдельными каталогами.
