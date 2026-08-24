// Toán thuần cho Weight Panel — tách riêng khỏi component để test được không
// cần render (docs UI competition studio §9.2/§21.1).

import type { SearchOptions } from "../../types";

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

/* ------------------------------------------------------------------ */
/* Solo — chạy MỘT (hoặc vài) nhánh, tắt phần còn lại.
 *
 * Backend đã nhận `branches.<id>.enabled=false` từ lâu, nhưng UI chỉ có ô tích
 * từng nhánh: muốn "chỉ tìm theo ASR" phải bỏ tích tám nhánh khác rồi tích lại
 * đủ tám nhánh đó khi xong. Không ai làm vậy giữa lúc thi, nên trên thực tế
 * không ai cô lập được một engine để xem nó tự đứng thì ra gì.
 *
 * Bỏ solo GỠ HẲN cờ `enabled` chứ không gán `true`: nhánh phải quay về đúng
 * mặc định của server (có nhánh server tắt sẵn), và weight/top_k đã chỉnh tay
 * phải còn nguyên.                                                          */
/* ------------------------------------------------------------------ */


/** Nhánh nào SẼ chạy với options hiện tại. */
export function runningBranches(options: SearchOptions, allBranchIds: string[]): string[] {
  return allBranchIds.filter((id) => options.branches?.[id]?.enabled !== false);
}

/** Đang cô lập nhánh = có ít nhất một nhánh bị tắt tường minh. */
export function isSoloActive(options: SearchOptions, allBranchIds: string[]): boolean {
  return allBranchIds.some((id) => options.branches?.[id]?.enabled === false);
}

/** Đặt tập nhánh được chạy. `soloed` rỗng = bỏ solo, trả tất cả về mặc định. */
export function applySolo(
  options: SearchOptions,
  allBranchIds: string[],
  soloed: string[]
): SearchOptions {
  const wanted = new Set(soloed);
  const branches = { ...(options.branches ?? {}) };
  for (const id of allBranchIds) {
    const current = branches[id];
    if (wanted.size === 0) {
      if (current && "enabled" in current) {
        const { enabled: _dropped, ...rest } = current;
        if (Object.keys(rest).length > 0) branches[id] = rest;
        else delete branches[id];
      }
      continue;
    }
    branches[id] = { ...current, enabled: wanted.has(id) };
  }
  return { ...options, branches };
}

/** Bấm nút Solo của một nhánh.
 *
 *  Chưa solo  -> chỉ nhánh đó chạy.
 *  Đang solo  -> thêm/bớt nhánh đó khỏi tập đang chạy.
 *  Bớt tới rỗng, hoặc thêm tới đủ tất cả -> bỏ solo (tập rỗng thì truy vấn
 *  chắc chắn không ra gì, đó không phải điều người bấm muốn).
 */
export function toggleSolo(
  options: SearchOptions,
  allBranchIds: string[],
  branchId: string
): SearchOptions {
  if (!isSoloActive(options, allBranchIds)) {
    return applySolo(options, allBranchIds, [branchId]);
  }
  const running = runningBranches(options, allBranchIds);
  const next = running.includes(branchId)
    ? running.filter((id) => id !== branchId)
    : [...running, branchId];
  if (next.length === 0 || next.length === allBranchIds.length) {
    return applySolo(options, allBranchIds, []);
  }
  return applySolo(options, allBranchIds, next);
}
