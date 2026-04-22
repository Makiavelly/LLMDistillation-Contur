"""
Генерация датасета Countdown через Qwen3-8B (llama-server)
=============================================================
Запуск:
    python generate_dataset.py

Что делает:
    1. Берёт промпты из HuggingFaceTB/Countdown-Task-GOLD (all, 3-4 числа)
    2. Генерирует синтетические задачи с 5-6 числами
    3. Отправляет на llama-server параллельно (concurrent requests)
    4. Фильтрует только правильные ответы (брутфорс-валидация)
    5. Сохраняет в data/countdown_qwen3_8b.jsonl
"""

import os
import re
import ast
import json
import time
import random
import operator
import itertools
import requests
import concurrent.futures
from pathlib import Path
from collections import Counter
from datasets import load_dataset

# ─── Конфигурация ────────────────────────────────────────────────────────────

CFG = {
    # Сервер
    "server_url"     : "http://localhost:8080/completion",
    "n_workers"      : 8,       # параллельных запросов (не больше числа слотов сервера)
    "timeout"        : 120,     # секунд на один запрос

    # Генерация
    "n_samples"      : 5,       # попыток на один промпт (rejection sampling)
    "temperature"    : 0.6,
    "top_k"          : 20,
    "top_p"          : 0.95,
    "max_tokens"     : 1024,    # Qwen3-8B thinking может быть длинным

    # Данные
    "dataset_name"   : "HuggingFaceTB/Countdown-Task-GOLD",
    "dataset_config" : "all",
    "synth_5_count"  : 8000,   # синтетических задач с 5 числами
    "synth_6_count"  : 5000,   # синтетических задач с 6 числами

    # Выход
    "output_path"    : "./data/countdown_qwen3_8b.jsonl",
    "checkpoint_path": "./data/checkpoint.jsonl",  # промежуточное сохранение
    "checkpoint_every": 500,   # сохранять каждые N примеров

    "seed"           : 42,
}

random.seed(CFG["seed"])

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning "
    "process in the mind and then provide the user with the answer."
)

# ─── Брутфорс-солвер ─────────────────────────────────────────────────────────

def solve_countdown(nums, target):
    """Возвращает выражение-строку или None."""
    ops = [operator.add, operator.sub, operator.mul, operator.truediv]
    op_syms = {operator.add:'+', operator.sub:'-',
               operator.mul:'*', operator.truediv:'/'}

    def solve(numbers):
        if len(numbers) == 1:
            val, expr = numbers[0]
            if abs(val - target) < 1e-9 and abs(val - round(val)) < 1e-9:
                return expr
            return None
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                if i == j:
                    continue
                a_val, a_expr = numbers[i]
                b_val, b_expr = numbers[j]
                rest = [numbers[k] for k in range(len(numbers)) if k != i and k != j]
                for op in ops:
                    if op == operator.truediv and abs(b_val) < 1e-9:
                        continue
                    if op == operator.truediv and abs(a_val % b_val) > 1e-9:
                        continue
                    result = op(a_val, b_val)
                    sym = op_syms[op]
                    if op in (operator.mul, operator.truediv):
                        a_str = f"({a_expr})" if any(s in a_expr for s in ('+','-')) else a_expr
                        b_str = f"({b_expr})" if any(s in b_expr for s in ('+','-')) else b_expr
                    else:
                        a_str = a_expr
                        b_str = (f"({b_expr})"
                                 if op == operator.sub and any(s in b_expr for s in ('+','-'))
                                 else b_expr)
                    new_expr = f"{a_str} {sym} {b_str}"
                    found = solve(rest + [(result, new_expr)])
                    if found is not None:
                        return found
        return None

    return solve([(float(n), str(n)) for n in nums])


def validate_equation(eq_str, nums, target):
    """Проверяет уравнение по правилам submission."""
    try:
        allowed = set("0123456789 +-*/().")
        if not set(eq_str) <= allowed:
            return False
        nums_in_eq = [int(x) for x in re.findall(r'\d+', eq_str)]
        nums_avail = sorted(nums)
        for n in sorted(nums_in_eq):
            if n not in nums_avail:
                return False
            nums_avail.remove(n)
        result = eval(eq_str)
        return abs(result - target) < 1e-6
    except Exception:
        return False


def extract_equation(text):
    """Извлекает выражение из <answer>...</answer>."""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip()
    if '=' in raw:
        return raw.split('=')[0].strip()
    return raw

# ─── Генерация синтетических задач ───────────────────────────────────────────

def generate_synthetic_tasks(n_5, n_6):
    """Генерирует задачи с 5-6 числами у которых есть решение."""
    print(f"\nГенерируем синтетические задачи: {n_5} с 5 числами, {n_6} с 6 числами...")
    tasks = []
    for n_nums, count in [(5, n_5), (6, n_6)]:
        generated = 0
        attempts = 0
        while generated < count:
            attempts += 1
            nums = random.choices(range(1, 101), k=n_nums)
            target = random.randint(10, 999)
            solution = solve_countdown(nums, target)
            if solution is None:
                continue
            tasks.append({"target": target, "nums": nums})
            generated += 1
            if generated % 1000 == 0:
                print(f"  {n_nums} чисел: {generated}/{count} (попыток: {attempts})")
        print(f"  {n_nums} чисел: готово {generated} задач")
    random.shuffle(tasks)
    return tasks

# ─── Загрузка промптов из датасета ───────────────────────────────────────────

def load_dataset_tasks():
    """Загружает задачи из all сплита (только target + nums)."""
    print(f"\nЗагружаем {CFG['dataset_name']} [{CFG['dataset_config']}]...")
    ds = load_dataset(CFG["dataset_name"], CFG["dataset_config"], split="train")
    tasks = [{"target": ex["target"], "nums": list(ex["nums"])} for ex in ds]
    print(f"  Загружено: {len(tasks)} задач")
    dist = Counter(len(t["nums"]) for t in tasks)
    for k in sorted(dist):
        print(f"    {k} чисел: {dist[k]} ({dist[k]/len(tasks)*100:.1f}%)")
    return tasks

# ─── Формирование промпта ─────────────────────────────────────────────────────

def make_user_prompt(nums, target):
    return (
        f"Using the numbers {nums}, create an equation that equals {target}. "
        f"You can use basic arithmetic operations (+, -, *, /) "
        f"and each number can only be used once. Show your work in "
        f"<think> </think> tags. And return the final equation and answer "
        f"in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    )


def make_messages(nums, target):
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": make_user_prompt(nums, target)},
    ]

# ─── Запрос к серверу ─────────────────────────────────────────────────────────

def query_server(nums, target):
    """
    Делает n_samples запросов к llama-server и возвращает
    первый правильный ответ или None.
    """
    # Формируем промпт в формате ChatML (Qwen3)
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{make_user_prompt(nums, target)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    for attempt in range(CFG["n_samples"]):
        try:
            resp = requests.post(
                CFG["server_url"],
                json={
                    "prompt"      : prompt,
                    "n_predict"   : CFG["max_tokens"],
                    "temperature" : CFG["temperature"],
                    "top_k"       : CFG["top_k"],
                    "top_p"       : CFG["top_p"],
                    "stop"        : ["<|im_end|>", "<|endoftext|>"],
                    "cache_prompt": False,
                },
                timeout=CFG["timeout"],
            )
            generated = resp.json().get("content", "")
            eq = extract_equation(generated)
            if eq and validate_equation(eq, nums, target):
                # Собираем полный ответ ассистента
                assistant_content = generated.strip()
                return {
                    "target"  : target,
                    "nums"    : nums,
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": make_user_prompt(nums, target)},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
        except Exception as e:
            if attempt == CFG["n_samples"] - 1:
                pass  # тихо пропускаем
    return None  # все попытки провалились

# ─── Основной цикл генерации ──────────────────────────────────────────────────

def run_generation(tasks):
    output_path = Path(CFG["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(CFG["checkpoint_path"])

    # Загружаем уже сгенерированные (resume)
    done_keys = set()
    results = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                ex = json.loads(line)
                done_keys.add((ex["target"], tuple(sorted(ex["nums"]))))
                results.append(ex)
        print(f"\nResume: уже есть {len(results)} примеров в чекпоинте")

    # Фильтруем уже сделанные
    tasks_todo = [
        t for t in tasks
        if (t["target"], tuple(sorted(t["nums"]))) not in done_keys
    ]
    print(f"Осталось задач: {len(tasks_todo)}")

    n_correct = len(results)
    n_failed  = 0
    start     = time.time()

    with open(checkpoint_path, "a") as ckpt_f:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=CFG["n_workers"]
        ) as executor:
            futures = {
                executor.submit(query_server, t["nums"], t["target"]): t
                for t in tasks_todo
            }
            for i, future in enumerate(
                concurrent.futures.as_completed(futures), 1
            ):
                result = future.result()
                if result is not None:
                    n_correct += 1
                    results.append(result)
                    ckpt_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if n_correct % CFG["checkpoint_every"] == 0:
                        ckpt_f.flush()
                else:
                    n_failed += 1

                # Прогресс
                #if i % 100 == 0 or i == len(futures):
                elapsed = time.time() - start
                speed   = i / elapsed if elapsed > 0 else 0
                eta     = (len(futures) - i) / speed if speed > 0 else 0
                rate    = n_correct / (n_correct + n_failed) * 100 if (n_correct + n_failed) > 0 else 0
                print(
                    f"  [{i:>6}/{len(futures)}] "
                    f"correct={n_correct} "
                    f"failed={n_failed} "
                    f"rate={rate:.1f}% "
                    f"speed={speed:.1f}/s "
                    f"ETA={eta/60:.1f}m"
                )

    # Финальное сохранение
    print(f"\nСохраняем {len(results)} примеров в {output_path}...")
    with open(output_path, "w") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Статистика
    dist = Counter(len(ex["nums"]) for ex in results)
    print(f"\n{'='*50}")
    print(f"ИТОГО: {len(results)} примеров")
    print(f"Распределение по длинам:")
    for k in sorted(dist):
        print(f"  {k} чисел: {dist[k]:>6} ({dist[k]/len(results)*100:.1f}%)")
    print(f"{'='*50}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("ГЕНЕРАЦИЯ ДАТАСЕТА COUNTDOWN — Qwen3-8B")
    print("="*60)

    # 1. Загружаем задачи из датасета (3-4 числа)
    dataset_tasks = load_dataset_tasks()

    # 2. Генерируем синтетические задачи (5-6 чисел)
    synth_tasks = generate_synthetic_tasks(
        CFG["synth_5_count"],
        CFG["synth_6_count"],
    )

    # 3. Объединяем и перемешиваем
    all_tasks = dataset_tasks + synth_tasks
    random.shuffle(all_tasks)

    dist = Counter(len(t["nums"]) for t in all_tasks)
    print(f"\nВсего задач для генерации: {len(all_tasks)}")
    for k in sorted(dist):
        print(f"  {k} чисел: {dist[k]} ({dist[k]/len(all_tasks)*100:.1f}%)")

    # 4. Запускаем генерацию
    run_generation(all_tasks)
