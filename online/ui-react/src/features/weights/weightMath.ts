// Toán thuần cho Weight Panel — tách riêng khỏi component để test được không
// cần render (docs UI competition studio §9.2/§21.1).

export interface WeightControlValue {
  branchId: string;
  label: string;
  enabled: boolean;
  weight: number;
  locked: boolean;
  available: boolean;
  unavailableReason?: string;
  colorToken: string;
}

function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

/** Chuẩn hoá để tổng các branch đang enabled = 1.0. Branch locked giữ nguyên
 * giá trị; phần còn lại (1 - tổng locked) chia cho các branch unlocked theo
 * đúng tỷ lệ hiện tại của chúng (không phải chia đều), trừ khi tổng unlocked
 * hiện tại là 0 thì chia đều.
 *
 * Ném lỗi nếu tổng trọng số bị khóa > 1.0 — không thể chuẩn hoá phần còn lại
 * về một số âm. */
export function normalizeWeights(items: WeightControlValue[]): WeightControlValue[] {
  const enabled = items.filter((x) => x.enabled && x.available);
  const locked = enabled.filter((x) => x.locked);
  const unlocked = enabled.filter((x) => !x.locked);

  const lockedSum = locked.reduce((s, x) => s + x.weight, 0);
  if (lockedSum > 1) {
    throw new Error("Tổng trọng số bị khóa vượt quá 1.0");
  }

  const remaining = 1 - lockedSum;
  const unlockedSum = unlocked.reduce((s, x) => s + x.weight, 0);

  return items.map((item) => {
    if (!item.enabled || !item.available || item.locked) return item;

    const normalized =
      unlockedSum > 0 ? (item.weight / unlockedSum) * remaining : remaining / Math.max(unlocked.length, 1);

    return { ...item, weight: round4(normalized) };
  });
}

/** Chuẩn hoá MỘT cột của bảng per-step weights (mỗi step tự chuẩn hoá riêng,
 * không liên quan tới step khác). */
export function normalizeStepColumn(column: Record<string, number>): Record<string, number> {
  const sum = Object.values(column).reduce((s, v) => s + v, 0);
  if (sum <= 0) return column;
  const result: Record<string, number> = {};
  for (const [key, value] of Object.entries(column)) {
    result[key] = round4(value / sum);
  }
  return result;
}
