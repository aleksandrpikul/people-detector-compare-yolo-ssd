# crowd-detector-yolo-vs-ssd

Сравнение двух нейросетевых алгоритмов детекции людей на видео `crowd.mp4` с единым кросс-платформенным пайплайном инференса.

Реализованы две разные архитектуры:
- **YOLOv8 (ONNX)** — современный одностадийный детектор, хорошо справляется со сценами с толпой и перекрытиями.
- **SSD MobileNet (ONNX + ONNXRuntime)** — компактный и быстрый детектор, но обычно хуже по полноте детекции в плотных сценах.

На выходе для каждой модели формируется:
- видео `*.mp4` с bbox вокруг людей и значением **confidence**
- файл метрик `*.metrics.json` (скорость/латентность/количество детекций)

---

## YOLOv8 (видео)

<video src="crowd_yolov8_test.mp4" controls width="900"></video>

## SSD MobileNet (видео)

<video src="crowd_ssd_test.mp4" controls width="900"></video>


---

## Содержимое репозитория

- `src/` — исходный код
  - `main.py` — CLI-запуск
  - `pipeline.py` — общий пайплайн чтения/отрисовки/сохранения видео
  - `download_models.py` — загрузка предобученных весов
  - `detectors/` — реализации детекторов (YOLOv8 и SSD)
- `models/` — скачанные веса ONNX (создаётся автоматически)
- `outputs/` — выходные видео + метрики (создаётся автоматически)
- `reports/` — отчёт и шаблоны
- `requirements.txt` — зависимости

---

## Установка

### Вариант 1: локально (Linux / MacOS / Windows)

1) Установить зависимости:

```bash
pip install -r requirements.txt
````

2. Скачать веса моделей:

```bash
python -m src.download_models
```

---

### Вариант 2: Google Colab

1. Загрузить в Colab:

* архив репозитория (zip)
* файл `crowd.mp4`

2. Распаковать и установить зависимости:

```bash
unzip -o crowd-detector-yolo-vs-ssd.zip -d crowd-detector-yolo-vs-ssd
cd crowd-detector-yolo-vs-ssd
pip install -r requirements.txt
python -m src.download_models
```

---

## Запуск инференса

### 1) YOLOv8

```bash
python -m src.main --input crowd.mp4 --model yolov8 --output outputs/crowd_yolov8.mp4
```

Результаты:

* `outputs/crowd_yolov8.mp4`
* `outputs/crowd_yolov8.mp4.metrics.json`

---

### 2) SSD MobileNet

```bash
python -m src.main --input crowd.mp4 --model ssd --output outputs/crowd_ssd.mp4
```

Результаты:

* `outputs/crowd_ssd.mp4`
* `outputs/crowd_ssd.mp4.metrics.json`

---

## Как посмотреть результаты в Colab

### Просмотр видео в ноутбуке (YOLO + SSD)

````python
import json
from pathlib import Path
from base64 import b64encode
from IPython.display import HTML, display, Markdown

def show_video(path, width=900):
    path = str(path)
    if not Path(path).exists():
        display(Markdown(f"Не найден файл: `{path}`"))
        return
    mp4 = Path(path).read_bytes()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML(f"""
    <video width="{width}" controls style="border:1px solid #ddd; border-radius:10px;">
      <source src="{data_url}" type="video/mp4">
    </video>
    """))

def show_metrics(metrics_path):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        display(Markdown(f"Метрики не найдены: `{metrics_path}`"))
        return None
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    pretty = {
        k: m[k] for k in [
            "model", "frames", "seconds_total", "avg_fps",
            "p50_ms_per_frame", "p95_ms_per_frame",
            "avg_people_per_frame", "avg_score"
        ] if k in m
    }
    display(Markdown("**Метрики:**\n```json\n" + json.dumps(pretty, ensure_ascii=False, indent=2) + "\n```"))
    return m

display(Markdown("## YOLOv8"))
show_video("outputs/crowd_yolov8.mp4")
show_metrics("outputs/crowd_yolov8.mp4.metrics.json")

display(Markdown("---\n## SSD MobileNet"))
show_video("outputs/crowd_ssd.mp4")
show_metrics("outputs/crowd_ssd.mp4.metrics.json")
````

---

### Быстрый просмотр через ссылки 

```python
from IPython.display import FileLink, display
display(FileLink("outputs/crowd_yolov8.mp4"))
display(FileLink("outputs/crowd_ssd.mp4"))
```

---

### Скачать результаты из Colab

```python
from google.colab import files
files.download("outputs/crowd_yolov8.mp4")
files.download("outputs/crowd_ssd.mp4")
files.download("outputs/crowd_yolov8.mp4.metrics.json")
files.download("outputs/crowd_ssd.mp4.metrics.json")
```

---

## Метрики и интерпретация

После каждого запуска создаётся `*.metrics.json`, содержащий основные показатели:

* `seconds_total` — общее время обработки видео
* `avg_fps` — средний FPS
* `p50_ms_per_frame` / `p95_ms_per_frame` — медианная и 95-перцентиль задержки на кадр
* `avg_people_per_frame` — среднее число найденных людей на кадр (proxy-качество)
* `avg_score` — средняя уверенность (confidence) по детекциям

Чтение метрик:

```bash
cat outputs/crowd_yolov8.mp4.metrics.json
cat outputs/crowd_ssd.mp4.metrics.json
```

---

## Отчёт

Отчёт находится в `reports/`.

* `reports/report.md` — шаблон отчёта
* `reports/report_filled.md` — может быть сгенерирован автоматически (если используется `make_report.py`)

Генерация:

```bash
python -m src.make_report
```

---

## Выводы 

### Качество детекции на сценах с толпой

Для видео с большим числом людей и перекрытиями **YOLOv8** обычно:

* находит больше людей на дальнем плане,
* лучше держит перекрытия,
* даёт более стабильные bbox по времени.

**SSD MobileNet** часто:

* пропускает мелких людей и людей в глубине кадра,
* хуже работает при сильных перекрытиях.

### Производительность

* SSD MobileNet может быть быстрее на CPU, но зачастую проигрывает по качеству.
* YOLOv8 часто даёт лучший баланс качество/скорость для практического применения.

Итоговый выбор делается по совокупности:

* визуальная проверка роликов,
* `avg_fps`, `p50/p95_ms_per_frame`,
* `avg_people_per_frame`.

---

## Рекомендации по улучшению качества и скорости

### Улучшение качества

1. Настройка порогов `conf` и параметров NMS под конкретное видео.
2. Увеличение входного разрешения инференса для лучшей детекции мелких людей (ценой скорости).
3. Добавление трекинга (SORT/ByteTrack) для стабилизации рамок и уменьшения пропусков.
4. Дообучение/тонкая настройка на похожих данных (если доступна разметка).

### Ускорение

1. Аппаратное ускорение: CUDA / TensorRT / OpenVINO.
2. Квантизация (FP16/INT8) и оптимизация ONNX-графа.
3. Параллелизация: декодирование видео и инференс в разных потоках.
4. Детекция не на каждом кадре + трекинг между кадрами.

---
