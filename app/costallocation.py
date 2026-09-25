"""已保存方案的烧成费用分摊：按窑次独立按器物体积比例分费。

请求按窑次给出每窑的烧成费用（分，非负整数），并按器物编号给出每件的
收费上限（分，非负整数）；两组均须完整覆盖原方案。每窑独立分摊，
器物份额为 ``min(上限, λ × 体积)``，其中 λ 用**精确有理数**求解，使该窑
全部器物份额之和恰等于该窑费用；费用超过该窑上限总和时无法分摊。

λ 的求法（同窑器物至多 14 件）：把器物按阈值 ``t = 上限 / 体积`` 升序
分组。``S(λ) = Σ min(cᵢ, λ·vᵢ)`` 是关于 λ 的单调不减连续分段线性函数；
逐组推进时，尚未达到阈值的器物按线性计费、已越过阈值的器物按上限计费，
在费用首次不低于 ``S(t)`` 的分段上用一次分数除法即可定出 λ；费用恰等于
上限总和时 λ 取最高阈值。λ ≥ 0（费用为非负整数，``S(0) = 0``）。

取整：各器物精确份额（``Fraction``）**向下取整**为整数分，取整产生的
剩余分按「小数余数降序、器物编号升序」逐件补 1 分，直至分完。分摊后
各窑整数份额之和必等于请求费用。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction


class KilnCapInsufficientError(ValueError):
    """某窑烧成费用超过该窑器物收费上限总和，无法分摊。"""

    def __init__(self, kiln_index: int, cost_fen: int, cap_sum_fen: int) -> None:
        self.kiln_index = kiln_index
        self.cost_fen = cost_fen
        self.cap_sum_fen = cap_sum_fen
        super().__init__(
            f"窑次 {kiln_index} 的烧成费用 {cost_fen} 分超过该窑器物"
            f"收费上限总和 {cap_sum_fen} 分"
        )


@dataclass(frozen=True)
class ItemCostAllocation:
    """单件器物在其所属窑内的费用分摊。"""

    item_id: str
    kiln_index: int
    volume: int
    cap_fen: int                       # 请求的收费上限（分）
    exact_share: Fraction              # 精确份额 min(上限, λ×体积)（分）
    allocated_fen: int                 # 取整补齐后的整数份额（分）


@dataclass(frozen=True)
class KilnCostAllocation:
    """单个窑次的费用分摊结果。"""

    kiln_index: int
    cost_fen: int                      # 请求的该窑费用（分）
    cap_sum_fen: int                   # 该窑器物收费上限总和（分）
    lambda_value: Fraction             # 精确比例系数 λ（分 / 体积）
    items: tuple[ItemCostAllocation, ...]  # 按器物编号升序
    allocated_total_fen: int           # 各器物整数份额之和，等于 cost_fen


@dataclass(frozen=True)
class PlanCostAllocation:
    """整份方案各窑独立分摊后的结果。"""

    total_cost_fen: int
    total_cap_sum_fen: int
    kilns: tuple[KilnCostAllocation, ...]  # 按窑次序号升序


def _solve_lambda(
    members: Sequence[tuple[str, int]],
    caps: Mapping[str, int],
    cost: int,
) -> Fraction:
    """求使 Σ min(cᵢ, λ·vᵢ) = cost 的精确 λ（λ ≥ 0）。

    调用前须保证 0 ≤ cost ≤ Σcᵢ。members 为 (器物编号, 体积)，
    体积均为正整数。
    """

    # 按阈值 t = 上限 / 体积 升序分组（阈值相同的器物同时被截断）。
    groups: dict[Fraction, tuple[int, int]] = {}
    for item_id, volume in members:
        t = Fraction(caps[item_id], volume)
        vol, cap = groups.get(t, (0, 0))
        groups[t] = (vol + volume, cap + caps[item_id])

    capped_sum = Fraction(0)   # 阈值严格小于当前段的器物上限之和
    active_vol = sum(v for _, v in members)  # 尚未越过阈值的器物体积之和
    target = Fraction(cost)

    for t in sorted(groups):
        # 当前分段右端点 t 处的份额和（连续函数，开段 / 闭段取值相同）：
        # 已截断器物取上限，其余取 t×体积；阈值为 0 的器物自然包含在内。
        s_at_t = capped_sum + t * active_vol
        if target <= s_at_t:
            # λ 落在上一阈值与 t 之间（恰为 t 时本分组也恰好截断）。
            return (target - capped_sum) / active_vol
        group_vol, group_cap = groups[t]
        capped_sum += group_cap
        active_vol -= group_vol

    # cost ≤ 上限总和已在调用前保证，故必在某个分段内取到，不可达。
    raise AssertionError("费用分摊 λ 求解失败：费用超出收费上限总和")


def allocate_kiln_cost(
    kiln_index: int,
    members: Sequence[tuple[str, int]],
    caps: Mapping[str, int],
    cost_fen: int,
) -> KilnCostAllocation:
    """分摊单个窑次的烧成费用。

    :param kiln_index: 窑次序号（从 1 开始，原样回填）
    :param members: (器物编号, 正整数体积) 序列，即该窑装入的全部器物
    :param caps: 器物编号 -> 收费上限（分，非负整数），须覆盖 members
    :param cost_fen: 该窑烧成费用（分，非负整数）
    :raises KilnCapInsufficientError: 费用超过该窑器物收费上限总和
    :raises ValueError: 参数非法（空窑 / 体积非正 / 费用或上限为负等）
    """

    if not members:
        raise ValueError(f"窑次 {kiln_index} 无器物，无法分摊费用")
    for item_id, volume in members:
        if not isinstance(volume, int) or volume < 1:
            raise ValueError(f"器物 {item_id} 体积须为正整数")
        if not isinstance(caps[item_id], int) or caps[item_id] < 0:
            raise ValueError(f"器物 {item_id} 的收费上限须为非负整数（分）")
    if not isinstance(cost_fen, int) or cost_fen < 0:
        raise ValueError(f"窑次 {kiln_index} 的烧成费用须为非负整数（分）")

    cap_sum = sum(caps[i] for i, _ in members)
    if cost_fen > cap_sum:
        raise KilnCapInsufficientError(kiln_index, cost_fen, cap_sum)

    lam = _solve_lambda(members, caps, cost_fen)

    # 精确份额 -> 向下取整，记录小数余数。
    floors: dict[str, int] = {}
    remainders: dict[str, Fraction] = {}
    for item_id, volume in members:
        exact = min(Fraction(caps[item_id]), lam * volume)
        floors[item_id] = exact.numerator // exact.denominator
        remainders[item_id] = exact - floors[item_id]

    # 取整缺口 = 费用 - 份额地板和（等于各余数之和，为小于件数的非负整数）；
    # 按余数降序、编号升序逐件补 1 分。
    remainder_fee = cost_fen - sum(floors.values())
    bonus_order = sorted(members, key=lambda m: (-remainders[m[0]], m[0]))
    bonuses = {item_id for item_id, _ in bonus_order[:remainder_fee]}

    items = tuple(
        ItemCostAllocation(
            item_id=item_id,
            kiln_index=kiln_index,
            volume=volume,
            cap_fen=caps[item_id],
            exact_share=min(Fraction(caps[item_id]), lam * volume),
            allocated_fen=floors[item_id] + (1 if item_id in bonuses else 0),
        )
        for item_id, volume in sorted(members, key=lambda m: m[0])
    )
    allocated_total = sum(it.allocated_fen for it in items)
    assert allocated_total == cost_fen

    return KilnCostAllocation(
        kiln_index=kiln_index,
        cost_fen=cost_fen,
        cap_sum_fen=cap_sum,
        lambda_value=lam,
        items=items,
        allocated_total_fen=allocated_total,
    )


def allocate_plan_cost(
    kilns: Sequence[tuple[int, Sequence[tuple[str, int]]]],
    item_caps: Mapping[str, int],
    kiln_costs: Mapping[int, int],
) -> PlanCostAllocation:
    """对整份方案逐窑独立分摊烧成费用。

    :param kilns: (窑次序号, (器物编号, 体积) 序列) 序列，窑次与器物的
        完备性由调用方（路由）校验
    :param item_caps: 器物编号 -> 收费上限（分，非负整数）
    :param kiln_costs: 窑次序号 -> 烧成费用（分，非负整数）
    :returns: 各窑分摊结果（按窑次序号升序）
    :raises KilnCapInsufficientError: 任一窑费用超过其器物上限总和时抛出
        首个超限窑的错误（各窑相互独立，错误信息只涉及该窑）
    """

    results: list[KilnCostAllocation] = []
    for kiln_index, members in sorted(kilns, key=lambda k: k[0]):
        results.append(
            allocate_kiln_cost(
                kiln_index,
                members,
                item_caps,
                kiln_costs[kiln_index],
            )
        )

    return PlanCostAllocation(
        total_cost_fen=sum(k.cost_fen for k in results),
        total_cap_sum_fen=sum(k.cap_sum_fen for k in results),
        kilns=tuple(results),
    )
