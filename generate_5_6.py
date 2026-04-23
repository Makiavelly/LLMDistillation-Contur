import os
import re
import ast
import json
import time
import random
import operator
import itertools
import requests
from pathlib import Path
from collections import Counter
from multiprocessing import Pool

random.seed(42)


# ─── Генерация синтетических задач ───────────────────────────────────────────

def _generate_one(args):
    """Выполняется в отдельном процессе."""
    import random, operator
    n_nums, seed = args
    random.seed(seed)
    
    ops = [operator.add, operator.sub, operator.mul, operator.truediv]
    op_syms = {operator.add:'+', operator.sub:'-',
               operator.mul:'*', operator.truediv:'/'}

    def solve(numbers, target):
        if len(numbers) == 1:
            val, expr = numbers[0]
            if abs(val - target) < 1e-9 and abs(val - round(val)) < 1e-9:
                return expr
            return None
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                if i == j: continue
                a_val, a_expr = numbers[i]
                b_val, b_expr = numbers[j]
                rest = [numbers[k] for k in range(len(numbers)) if k != i and k != j]
                for op in ops:
                    if op == operator.truediv and abs(b_val) < 1e-9: continue
                    if op == operator.truediv and abs(a_val % b_val) > 1e-9: continue
                    result = op(a_val, b_val)
                    sym = op_syms[op]
                    if op in (operator.mul, operator.truediv):
                        a_str = f"({a_expr})" if any(s in a_expr for s in ('+','-')) else a_expr
                        b_str = f"({b_expr})" if any(s in b_expr for s in ('+','-')) else b_expr
                    else:
                        a_str = a_expr
                        b_str = (f"({b_expr})" if op == operator.sub
                                 and any(s in b_expr for s in ('+','-')) else b_expr)
                    found = solve(rest + [(result, f"{a_str} {sym} {b_str}")], target)
                    if found is not None:
                        return found
        return None

    while True:
        nums = random.choices(range(1, 101), k=n_nums)
        target = random.randint(1, 999)
        solution = solve([(float(n), str(n)) for n in nums], target)
        if solution is not None:
            return {"target": target, "nums": nums}

def generate_synthetic_tasks(n_5, n_6):
    synth_path = Path("./data/synthetic_5_6.jsonl")
    
    # Если уже сгенерировано — загружаем
    if synth_path.exists():
        print(f"\nНайден файл синтетики: {synth_path}")
        with open(synth_path) as f:
            results = [json.loads(line) for line in f]
        print(f"  Загружено: {len(results)} примеров")
        dist = Counter(len(r["nums"]) for r in results)
        for k in sorted(dist):
            print(f"    {k} чисел: {dist[k]}")
        return results
    
    print(f"\nГенерируем синтетику: {n_5} задач с 5 числами, {n_6} с 6...")
    
    args = []
    for i in range(n_5):
        args.append((5, random.randint(0, 10**9)))
    for _ in range(n_6):
        args.append((6, random.randint(0, 10**9)))
        
    
    random.shuffle(args)
    
    with Pool(processes=12) as pool:
        results = []
        for i, result in enumerate(
            pool.imap_unordered(_generate_one, args, chunksize=50), 1
        ):
            results.append(result)
            if i % 100 == 0 or i == len(args):
                print(f"  [{i}/{len(args)}] готово")
                
    random.shuffle(results)
    print(f"  Готово: {len(results)} задач")
    
    # Сохраняем синтетику отдельно
    import json
    synth_path = "./data/synthetic_5_6.jsonl"
    Path(synth_path).parent.mkdir(parents=True, exist_ok=True)
    with open(synth_path, "w") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Синтетика сохранена: {synth_path} ({len(results)} примеров)")
    
    return results

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    synth_tasks = generate_synthetic_tasks(
        21000,
        11000,
    )
