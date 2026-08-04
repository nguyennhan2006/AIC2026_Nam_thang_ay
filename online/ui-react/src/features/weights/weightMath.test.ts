import { describe, expect, it } from "vitest";
import { normalizeStepColumn, normalizeWeights } from "./weightMath";
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
