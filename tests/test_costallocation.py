"""costallocation 单元测试：固定用例 + 随机用例的独立参考实现交叉验证。"""

import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.costallocation import (  # noqa: E402
    KilnCapInsufficientError,
    allocate_kiln_cost,
    allocate_plan_cost,
)


def _reference_lambda(members, caps, cost):
    """独立参考实现：枚举「截断 / 线性」器物划分反解 λ，再验证分段条件。

    设线性器物集合 A（份额 = λv）、截断器物集合 B（份额 = 上限），则
    λ = (cost − Σ_B c) / Σ_A v；解合法当且仅当 A 中 λv ≤ c、B 中 λv ≥ c。
    合法划分给出的 λ 必唯一（S 连续单调）。
    """

    if cost == 0:
        return Fraction(0)
    n = len(members)
    candidates = set()
    for mask in range(1 << n):
        linear = [members[k] for k in range(n) if not (mask >> k) & 1]
        capped = [members[k] for k in range(n) if (mask >> k) & 1]
        lin_vol = sum(v for _, v in linear)
        cap_sum_b = sum(caps[i] for i, _ in capped)
        if lin_vol == 0:
            if cost != cap_sum_b:
                continue
            lam = max(Fraction(caps[i], v) for i, v in members)
        else:
            lam = Fraction(cost - cap_sum_b, lin_vol)
        if lam < 0:
            continue
        if all(lam * v <= caps[i] for i, v in linear) and all(
            lam * v >= caps[i] for i, v in capped
        ):
            candidates.add(lam)
    assert len(candidates) == 1, f"参考 λ 不唯一: {candidates}"
    return candidates.pop()


def _reference_allocate(members, caps, cost):
    """参考实现：λ + floor + 余数降序/编号升序补齐，返回 {编号: 整数份额}。"""

    cap_sum = sum(caps[i] for i, _ in members)
    if cost > cap_sum:
        return None
    lam = _reference_lambda(members, caps, cost)
    exact = {i: min(Fraction(caps[i]), lam * v) for i, v in members}
    floors = {i: e.numerator // e.denominator for i, e in exact.items()}
    rem = cost - sum(floors.values())
    order = sorted(members, key=lambda m: (-(exact[m[0]] - floors[m[0]]), m[0]))
    for i, _ in order[:rem]:
        floors[i] += 1
    assert sum(floors.values()) == cost
    return floors


def _check_invariants(result, members, caps, cost):
    """校验分摊结果的全部不变式。"""

    assert {it.item_id for it in result.items} == {i for i, _ in members}
    assert result.allocated_total_fen == cost
    total = 0
    for it in result.items:
        total += it.allocated_fen
        assert 0 <= it.allocated_fen <= caps[it.item_id]
        assert it.allocated_fen <= it.exact_share + 1
        assert it.allocated_fen - 1 < it.exact_share
        assert it.exact_share == min(Fraction(caps[it.item_id]),
                                    result.lambda_value * it.volume)
    assert total == cost
    exact_sum = sum(it.exact_share for it in result.items)
    assert exact_sum == cost
    # λ ≥ 0
    assert result.lambda_value >= 0


def test_proportional_exact():
    # 体积 2/3/5、费用 20、上限充足：比例恰好整除
    members = [("p1", 2), ("p2", 3), ("p3", 5)]
    caps = {"p1": 1000, "p2": 1000, "p3": 1000}
    r = allocate_kiln_cost(1, members, caps, 20)
    assert [it.allocated_fen for it in r.items] == [4, 6, 10]
    assert r.lambda_value == 2
    _check_invariants(r, members, caps, 20)


def test_floor_and_remainder_order():
    # 体积 1/2/3、费用 10：λ=5/3，份额 5/3、10/3、5 -> 地板 1,3,5，余 1
    # 余数：a=2/3 > b=1/3，补 1 分给编号更小且余数更大的 a
    members = [("a", 1), ("b", 2), ("c", 3)]
    caps = {"a": 100, "b": 100, "c": 100}
    r = allocate_kiln_cost(7, members, caps, 10)
    assert [it.allocated_fen for it in r.items] == [2, 3, 5]
    assert r.lambda_value == Fraction(5, 3)
    _check_invariants(r, members, caps, 10)


def test_equal_volume_tie_by_id():
    # 等体积、费用 10：精确份额各 10/3，余数相同，按编号升序补 a、b
    members = [("a", 1), ("b", 1), ("c", 1)]
    caps = {"a": 100, "b": 100, "c": 100}
    r = allocate_kiln_cost(6, members, caps, 10)
    assert [it.allocated_fen for it in r.items] == [4, 3, 3]
    _check_invariants(r, members, caps, 10)


def test_cap_binding():
    # p3 上限 3 先被截断，费用 13 -> p1=4、p2=6、p3=3，λ=2
    members = [("p1", 2), ("p2", 3), ("p3", 5)]
    caps = {"p1": 1000, "p2": 1000, "p3": 3}
    r = allocate_kiln_cost(2, members, caps, 13)
    assert [it.allocated_fen for it in r.items] == [4, 6, 3]
    assert r.lambda_value == 2
    assert r.cap_sum_fen == 2003
    _check_invariants(r, members, caps, 13)


def test_zero_cost():
    # 费用 0：λ=0，全部份额为 0（含 0 上限器物）
    members = [("p1", 2), ("p2", 3)]
    caps = {"p1": 0, "p2": 5}
    r = allocate_kiln_cost(3, members, caps, 0)
    assert [it.allocated_fen for it in r.items] == [0, 0]
    assert r.lambda_value == 0
    _check_invariants(r, members, caps, 0)


def test_cost_equals_cap_sum():
    members = [("p1", 2), ("p2", 3)]
    caps = {"p1": 4, "p2": 6}
    r = allocate_kiln_cost(4, members, caps, 10)
    assert [it.allocated_fen for it in r.items] == [4, 6]
    assert r.lambda_value == 2
    _check_invariants(r, members, caps, 10)


def test_single_item():
    # 单件器物承担全部费用（不超过自身上限）
    members = [("p1", 2)]
    caps = {"p1": 7}
    r = allocate_kiln_cost(5, members, caps, 7)
    assert r.items[0].allocated_fen == 7
    assert r.lambda_value == Fraction(7, 2)
    _check_invariants(r, members, caps, 7)


def test_insufficient_cap():
    try:
        allocate_kiln_cost(8, [("p1", 2)], {"p1": 7}, 8)
        assert False, "应抛出 KilnCapInsufficientError"
    except KilnCapInsufficientError as e:
        assert e.kiln_index == 8 and e.cost_fen == 8 and e.cap_sum_fen == 7


def test_invalid_arguments():
    bad_calls = (
        lambda: allocate_kiln_cost(1, [], {}, 0),                  # 空窑
        lambda: allocate_kiln_cost(1, [("a", 0)], {"a": 1}, 0),   # 体积非正
        lambda: allocate_kiln_cost(1, [("a", 1)], {"a": -1}, 0),  # 上限负
        lambda: allocate_kiln_cost(1, [("a", 1)], {"a": 1}, -1),  # 费用负
    )
    for bad in bad_calls:
        try:
            bad()
            assert False, "应抛出 ValueError"
        except (ValueError, KeyError):
            pass


def test_random_cross_check():
    rng = random.Random(20260925)
    for trial in range(3000):
        n = rng.randint(1, 8)
        ids = [f"i{k}" for k in range(n)]
        members = [(i, rng.randint(1, 20)) for i in ids]
        caps = {i: rng.randint(0, 50) for i in ids}
        cap_sum = sum(caps.values())
        cost = rng.randint(0, cap_sum + 5)
        if cost > cap_sum:
            try:
                allocate_kiln_cost(1, members, caps, cost)
                assert False, f"trial {trial}: 应抛出上限不足错误"
            except KilnCapInsufficientError:
                continue
        got = allocate_kiln_cost(trial + 1, members, caps, cost)
        expect = _reference_allocate(members, caps, cost)
        got_map = {it.item_id: it.allocated_fen for it in got.items}
        assert got_map == expect, (
            f"trial {trial}: {got_map} != {expect}, "
            f"members={members}, caps={caps}, cost={cost}"
        )
        _check_invariants(got, members, caps, cost)


def test_plan_level_independent_kilns():
    # 多窑独立分摊：各窑合计 = 请求费用，总额一致；某窑超限即报错
    kilns = [
        (1, [("a", 1), ("b", 1)]),
        (2, [("c", 3)]),
        (3, [("d", 2), ("e", 3), ("f", 5)]),
    ]
    caps = {"a": 100, "b": 100, "c": 4, "d": 100, "e": 100, "f": 3}
    costs = {1: 10, 2: 4, 3: 13}
    plan = allocate_plan_cost(kilns, caps, costs)
    assert plan.total_cost_fen == 27
    assert plan.total_cap_sum_fen == 407
    totals = {}
    for k in plan.kilns:
        totals[k.kiln_index] = k.allocated_total_fen
        assert k.allocated_total_fen == costs[k.kiln_index]
    assert totals == {1: 10, 2: 4, 3: 13}

    # 窑 2 费用 5 > 器物 c 上限 4：明确报错并指明该窑
    try:
        allocate_plan_cost(kilns, caps, {1: 10, 2: 5, 3: 13})
        assert False, "应抛出 KilnCapInsufficientError"
    except KilnCapInsufficientError as e:
        assert e.kiln_index == 2 and e.cost_fen == 5 and e.cap_sum_fen == 4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n全部 {len(fns)} 个测试通过")
