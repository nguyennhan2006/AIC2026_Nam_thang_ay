import { describe, expect, it } from "vitest";
import { zoneForRank } from "./types";

describe("zoneForRank", () => {
  it("matches the official scoring cutoffs 1 / 5 / 20 / 50 / 100", () => {
    expect(zoneForRank(1)).toBe("rank_1");
    expect(zoneForRank(2)).toBe("ranks_2_5");
    expect(zoneForRank(5)).toBe("ranks_2_5");
    expect(zoneForRank(6)).toBe("ranks_6_20");
    expect(zoneForRank(20)).toBe("ranks_6_20");
    expect(zoneForRank(21)).toBe("ranks_21_50");
    expect(zoneForRank(50)).toBe("ranks_21_50");
    expect(zoneForRank(51)).toBe("ranks_51_100");
    expect(zoneForRank(100)).toBe("ranks_51_100");
  });

  it("returns beyond_100 past the submission limit", () => {
    expect(zoneForRank(101)).toBe("beyond_100");
    expect(zoneForRank(500)).toBe("beyond_100");
  });
});
