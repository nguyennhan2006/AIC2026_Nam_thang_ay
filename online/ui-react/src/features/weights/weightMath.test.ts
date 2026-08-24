import { describe, expect, it } from "vitest";
import {
  applySolo, isSoloActive, normalizeStepColumn, normalizeWeights, runningBranches, toggleSolo,
} from "./weightMath";
import type { WeightControlValue } from "./weightMath";

function item(overrides: Partial<WeightControlValue> = {}): WeightControlValue {
  return {
    branchId: "dense_visual",
    label: "Visual",
    enabled: true,
    weight: 1,
    locked: false,
    available: true,
    colorToken: "",
    ...overrides,
  };
}

describe("normalizeWeights", () => {
  it("normalizes unlocked weights so the enabled total is 1.0", () => {
    const items = [
      item({ branchId: "a", weight: 3 }),
      item({ branchId: "b", weight: 1 }),
    ];
    const result = normalizeWeights(items);
    const total = result.reduce((sum, x) => sum + x.weight, 0);
    expect(total).toBeCloseTo(1, 4);
    // tỷ lệ 3:1 giữ nguyên sau khi chuẩn hoá
    expect(result[0].weight / result[1].weight).toBeCloseTo(3, 4);
  });

  it("preserves locked weights exactly and only redistributes the remainder", () => {
    const items = [
      item({ branchId: "a", weight: 0.5, locked: true }),
      item({ branchId: "b", weight: 1 }),
      item({ branchId: "c", weight: 3 }),
    ];
    const result = normalizeWeights(items);
    const locked = result.find((x) => x.branchId === "a")!;
    expect(locked.weight).toBe(0.5);
    const unlockedTotal = result.filter((x) => !x.locked).reduce((sum, x) => sum + x.weight, 0);
    expect(unlockedTotal).toBeCloseTo(0.5, 4);
  });

  it("rejects a locked sum greater than 1.0", () => {
    const items = [item({ branchId: "a", weight: 0.7, locked: true }), item({ branchId: "b", weight: 0.6, locked: true })];
    expect(() => normalizeWeights(items)).toThrow();
  });

  it("skips disabled and unavailable branches entirely", () => {
    const items = [
      item({ branchId: "a", weight: 5 }),
      item({ branchId: "b", weight: 5, enabled: false }),
      item({ branchId: "c", weight: 5, available: false }),
    ];
    const result = normalizeWeights(items);
    expect(result.find((x) => x.branchId === "b")!.weight).toBe(5);
    expect(result.find((x) => x.branchId === "c")!.weight).toBe(5);
    expect(result.find((x) => x.branchId === "a")!.weight).toBeCloseTo(1, 4);
  });

  it("splits evenly when every unlocked branch starts at zero weight", () => {
    const items = [item({ branchId: "a", weight: 0 }), item({ branchId: "b", weight: 0 })];
    const result = normalizeWeights(items);
    expect(result[0].weight).toBeCloseTo(0.5, 4);
    expect(result[1].weight).toBeCloseTo(0.5, 4);
  });
});

describe("normalizeStepColumn", () => {
  it("normalizes one TRAKE step's modality weights to sum to 1.0", () => {
    const result = normalizeStepColumn({ visual: 2, ocr: 2 });
    expect(result.visual).toBeCloseTo(0.5, 4);
    expect(result.ocr).toBeCloseTo(0.5, 4);
  });

  it("returns the column unchanged when the sum is zero", () => {
    const column = { visual: 0, ocr: 0 };
    expect(normalizeStepColumn(column)).toEqual(column);
  });
});

describe("solo — cô lập một engine", () => {
  const ALL = ["dense_visual", "caption_bm25", "bm25_ocr", "bm25_asr"];

  it("chưa solo thì mọi nhánh đều chạy", () => {
    expect(isSoloActive({}, ALL)).toBe(false);
    expect(runningBranches({}, ALL)).toEqual(ALL);
  });

  it("bấm solo ASR: chỉ ASR chạy, các nhánh khác tắt tường minh", () => {
    const next = toggleSolo({}, ALL, "bm25_asr");
    expect(isSoloActive(next, ALL)).toBe(true);
    expect(runningBranches(next, ALL)).toEqual(["bm25_asr"]);
    expect(next.branches?.dense_visual?.enabled).toBe(false);
    expect(next.branches?.bm25_asr?.enabled).toBe(true);
  });

  it("solo thêm nhánh thứ hai thì chạy cả hai", () => {
    const asr = toggleSolo({}, ALL, "bm25_asr");
    const both = toggleSolo(asr, ALL, "caption_bm25");
    expect(runningBranches(both, ALL).sort()).toEqual(["bm25_asr", "caption_bm25"]);
  });

  it("bỏ solo nhánh cuối cùng = bỏ solo hẳn, không để tập rỗng", () => {
    const asr = toggleSolo({}, ALL, "bm25_asr");
    const cleared = toggleSolo(asr, ALL, "bm25_asr");
    expect(isSoloActive(cleared, ALL)).toBe(false);
    expect(runningBranches(cleared, ALL)).toEqual(ALL);
  });

  it("solo đủ hết mọi nhánh = bỏ solo, không giữ cờ enabled thừa", () => {
    let options = toggleSolo({}, ALL, "bm25_asr");
    for (const id of ALL.filter((x) => x !== "bm25_asr")) {
      options = toggleSolo(options, ALL, id);
    }
    expect(isSoloActive(options, ALL)).toBe(false);
    for (const id of ALL) expect(options.branches?.[id]?.enabled).toBeUndefined();
  });

  it("bỏ solo GIỮ NGUYÊN trọng số đã chỉnh tay", () => {
    const tuned = { branches: { dense_visual: { weight: 3 }, bm25_asr: { weight: 0.5 } } };
    const soloed = toggleSolo(tuned, ALL, "bm25_asr");
    const cleared = applySolo(soloed, ALL, []);
    expect(cleared.branches?.dense_visual?.weight).toBe(3);
    expect(cleared.branches?.bm25_asr?.weight).toBe(0.5);
    expect(cleared.branches?.dense_visual?.enabled).toBeUndefined();
  });

  it("không đụng tới nhánh nằm ngoài danh sách được phép solo", () => {
    // Nhánh server không đăng ký mà bị gán enabled sẽ làm /capabilities trả 422.
    const next = toggleSolo({ branches: { la_mat: { weight: 2 } } }, ALL, "bm25_ocr");
    expect(next.branches?.la_mat).toEqual({ weight: 2 });
  });
});
