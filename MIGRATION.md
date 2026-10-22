# TinySMS

Проект на основе [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template),
commit `600143298293fc2c17497e3becbf4f221d5506bb`.

## Установка и данные

```bash
# ПК; при необходимости сначала установить нужную CUDA-сборку torch.
pip install -r requirements.txt
# Colab/Kaggle: сохранить установленный платформой torch.
pip install -r requirements-notebook.txt
```

Пути в конфиге разрешаются относительно корня проекта:

```yaml
data:
  dataset_path: ../hw2/artifacts/tokenized_dataset
  tokenizer_path: artifacts/tokenizer.json
```

При переносе нужны исходный Arrow-датасет, токенизатор и папка
`artifacts/clean_numeric_1m/`. Текущий токенизатор: vocab=8192, PAD=0, BOS=2,
EOS=3. Collator добавляет отсутствующий EOS и делает динамический padding;
loss игнорирует PAD. Исходные Arrow-файлы не изменяются.

Подготовка нового токенизатора/датасета: `python prepare_data.py --config src/configs/tokenizer.yaml`.
Аудит длин и исполнимости: `notebooks/dataset_audit.ipynb`.

## Текущий эксперимент

```bash
python train.py -cn=tinysms_strong_clean
```

Обучение с нуля: encoder 6 / decoder 4, d_model=512, d_ff=2048, 8 heads,
dropout=0.1. Физический batch 32, accumulation 4 — эффективный batch 128.
AMP auto выбирает native BF16, иначе FP16 + GradScaler; на CPU используется
FP32. Включены fused AdamW и группировка по длине. Activation checkpointing
выключен; `+trainer.compile.enabled=true` включает компиляцию.

```yaml
trainer:
  epoch_len: 1000             # Микробатчей в условной эпохе.
  n_epochs: 93.756            # Всего 93756 микробатчей / 23439 обновлений.
  accumulate_grad_batches: 4
  pass_every_epochs: 10
  log_step: 100               # Ось логирования — микробатчи.
  early_stop: 0
```

Val loss считается после каждой эпохи; Pass@1 — раз в 10 эпох и после
финальной неполной эпохи. Данные непрерывно идут между условными эпохами,
перемешиваются после полного прохода выборки. При `epoch_len=null` длина эпохи
равна длине train dataloader. Отдельного режима `max_steps` нет.

Loss для накопления и валидации нормируется по целевым токенам без PAD.
Clipping, optimizer и scheduler выполняются один раз на группу микробатчей.
При FP16 overflow scheduler не продвигается. Неполная группа в конце эпохи
тоже обновляет веса. Валидация сохраняет RNG обучения и режим модели.

Warmup + cosine собран из PyTorch LinearLR, CosineAnnealingLR и SequentialLR.
В чистом профиле warmup длится 4 условные эпохи (1000 обновлений), LR растёт
от 3e-7 до 3e-4, затем падает до 3e-5. Бюджет равен 23439 обновлениям грязного
baseline; из-за других границ эпох это на 6 микробатчей больше прежнего.

## Очистка и сравнение

Числовая очистка уже подготовлена: **964857** строк оставлено, **35143**
исключено. Удаляются только примеры, чей код не возвращает конечное вещественное
число. Дедупликации, ограничения длин и исправления решений нет.

`prepare_clean_dataset.py` сверяет все строки сырого и токенизированного train,
проверяет fingerprint и токенизатор, использует готовый аудит и перепроверяет
редкие ошибки. Результат — индексы `train_indices.npy`, manifest и CSV исключений.
Отчёт: `report/clean_numeric_1m.json`. Повторная подготовка:

```bash
python prepare_clean_dataset.py
```

Нужен локальный HF-кеш и готовый аудит. `--allow-download` разрешает загрузку
сырого датасета; `--output` задаёт новую папку без перезаписи старой.

Validation неизменна: 2048 строк после первого миллиона, включая 66 нечисловых
эталонов, исключаемых из знаменателя Pass@1. Test из 1919 задач не имеет labels.
Грязный baseline: **1059/1982 = 53.43%**, `saved/strong_dirty_1m/`.
Старый val loss усреднялся по батчам, поэтому напрямую сравнивать его с новым
нельзя. Для грязного контроля с текущим расписанием:

```bash
python train.py -cn=tinysms_strong_clean \
  datasets.train.indices_path=null writer.run_name=strong_dirty_virtual_1m
```

## Checkpoints и проверка кода

Новый вариант со словарём BPE **16384**:

```bash
python prepare_data.py --config src/configs/tokenizer_full_16k.yaml
python filter_lengths.py --config src/configs/filter_length_128_16k.yaml
python train.py -cn=tinysms_strong_long_16k
```

Первая команда обучает BPE на 100000 случайных очищенных train-примеров без
validation, затем токенизирует весь очищенный train, validation и public test.
Артефакты: `artifacts/tokenizer_16k.json`, `artifacts/tokenized_full_clean_16k/`.
Четыре специальных токена сохраняют порядок PAD=0, UNK=1, BOS=2, EOS=3;
размер словаря модели — 16384. Модель обучается с нуля, с прежними accumulation,
inverse-sqrt scheduler и расписанием валидации. Результаты — в
`saved/strong_full_clean_sqrt_16k_len128/`.

В 16k-конфиге явно выбран BF16 и `trainer.max_time_hours=10`. Это накопленное
время обучения. Val loss считается каждую условную эпоху, Pass@1 — каждые 20
условных эпох (`trainer.pass_every_epochs=20`). Лимит учитывает
время внутри `trainer.train()`, включая загрузку батчей, validation, Pass@1 и
сохранения предыдущих checkpoints; начальная загрузка модели/датасета считается
отдельно. Лимит проверяется на границах условных эпох: текущая эпоха и её
валидация завершаются, затем записывается checkpoint. Поэтому завершение может
быть немного позднее 10 часов. `n_epochs=2000` — верхний предел; при текущей
скорости первым срабатывает ограничение времени. Счётчик `training_seconds`
сохраняется в checkpoint и учитывается при resume; перерыв между запусками
в бюджет не входит. Warmup и inverse-sqrt LR считаются по обновлениям весов.

Фильтр длины применяется только к train: отдельно не более 128 токенов вопроса
и 128 токенов решения с EOS. Остаётся **10609815** строк, исключается **760806**;
средняя длина решения с EOS после фильтра — **63.80** токена. Validation и все
1919 test-задач остаются целиком. Тексты и токены не обрезаются.

Список индексов и manifest сохраняются в `artifacts/length128_16k/` и подключены
через `datasets.train.indices_path`. Исходный Arrow-датасет не копируется и не
перезаписывается. Если 16k-датасет уже подготовлен, достаточно выполнить только
команду `filter_lengths.py`; повторная токенизация не нужна. Параметры фильтра
задаются в `src/configs/filter_length_128_16k.yaml`. Сравнение BF16/FP16 на одной
условной эпохе: `report/amp_epoch_len128_16k.md`. Бюджет теперь ограничивается
фактическим временем, а не оценкой количества эпох из этого замера.

Готовый аудит исполнения переиспользуется по fingerprint и размеру сырого
датасета: числовой результат кода не зависит от BPE. Длины для нового BPE
вычисляются заново; mean, median, p90 и доля решений до 60 токенов с EOS
выводятся после подготовки и сохраняются в `preparation.json` рядом с датасетом.
Если токенизатор уже обучен, для отдельного повтора токенизации выставить
`train_tokenizer: false`; существующий выходной датасет не перезаписывается.
Продолжение обучения:

```bash
python train.py -cn=tinysms_strong_long_16k trainer.resume_from=checkpoint-last.pth
```

Вариант со словарём BPE 8192:

Длинный запуск с нуля на полном числовом train. Сначала подготовить полный
датасет прежним токенизатором:

```bash
python prepare_data.py --config src/configs/tokenizer_full.yaml
```

Используются локальный HF-кеш и готовый аудит всех 11838003 строк. По проверенной
маске в train остаётся **11370621** строка, включая ранее восстановленную строку
183245. Прежние 2048 validation-строк исключены из train и сохранены отдельным
split `validation`; test из 1919 задач остаётся без labels. Токенизированные
splits сохраняются в `artifacts/tokenized_full_clean/`. Подготовка не обучает
модель и не переобучает токенизатор.

Затем запустить обучение:

```bash
python train.py -cn=tinysms_strong_long
```

530 условных эпох = 530000 микробатчей / 132500 обновлений, примерно 1.49 прохода
полного очищенного train. Старт с нуля, batch 32 × accumulation 4. Scheduler:
16 условных эпох warmup (4000 обновлений) до пикового LR=3e-4, затем спад
`LR = 3e-4 * sqrt(4000 / update)` без привязки к финальному шагу. Ось логирования
остаётся в микробатчах, val loss — каждую условную эпоху, Pass@1 — каждые 10
и в конце. Результаты — в `saved/strong_full_clean_sqrt/`. Бюджет рассчитан
примерно на 10 часов по предыдущим логам; распределение длин полного train,
скорость GPU и генерация влияют на время. Подготовка данных считается отдельно.
Продолжение после прерывания:

```bash
python train.py -cn=tinysms_strong_long trainer.resume_from=checkpoint-last.pth
```

Перед генерацией освобождаются logits teacher-forced forward. Каждый временный
validation-батч удаляется после обработки; по окончании validation сбрасывается
трекер, вызываются `gc.collect()` и `torch.cuda.empty_cache()` на CUDA. Градиенты
обучения освобождаются до validation. Внутри батчевого цикла CUDA-кеш не сбрасывается.

`checkpoint-last.pth` сохраняется после эпохи, `model_best.pth` — при улучшении
отслеживаемой метрики. Loss-only проверки не меняют лучший Pass@1 checkpoint.
Сохраняются optimizer, scheduler, scaler, RNG и позиция данных. Продолжение:

```bash
python train.py -cn=tinysms_strong_clean trainer.resume_from=checkpoint-last.pth
```

При прерывании используется последний записанный checkpoint. Для resume нужны
прежние данные, batching и расписание. Checkpoint прежнего цикла условных эпох
также поддерживается. Для загрузки только весов используется `trainer.from_pretrained=...`.

Pass@1 использует свободную greedy-генерацию до EOS. Каждый эталон и кандидат
запускается в отдельном процессе Bubblewrap с ограничением времени, памяти,
вывода и процессов; сеть и файлы проекта недоступны. Нужны Linux, `bwrap` и
user namespaces. На Colab/Kaggle изоляция может быть запрещена; тогда метрика
останавливает запуск, небезопасного fallback нет. CPU-проверки:

```bash
python -m unittest discover -s tests -v
```
