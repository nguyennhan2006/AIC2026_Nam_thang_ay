import { describe, expect, it } from "vitest";
import { buildSubmissionCsv, ExportValidationError, MAX_SUBMISSION_ROWS } from "./exportCsv";
import type { TrayItem } from "./types";

function item(overrides: Partial<TrayItem> = {}): TrayItem {
  return {
    scene_id: "L01_V001_S0002",
    video_id: "L01_V001",
    score: 0.5,
    best_keyframe_id: "L01_V001_S0002_F000750",
    best_timestamp_sec: 25.0,
    start_sec: 20.0,
    ...overrides,
  };
}

describe("buildSubmissionCsv", () => {
  it("formats KIS rows as '<video_id>, <frame_idx>' with no header", () => {
    const csv = buildSubmissionCsv({ task: "kis", items: [item()] });
    expect(csv).toBe("L01_V001, 750");
  });

  it("formats AVS rows identically to KIS (2 columns, no header)", () => {
    const csv = buildSubmissionCsv({ task: "avs", items: [item()] });
    expect(csv).toBe("L01_V001, 750");
  });

  it("joins multiple rows with a single newline, no trailing header", () => {
    const csv = buildSubmissionCsv({
      task: "kis",
      items: [item(), item({ video_id: "L02_V011", best_keyframe_id: "L02_V011_S0001_F001200" })],
    });
    expect(csv).toBe("L01_V001, 750\nL02_V011, 1200");
    expect(csv.split("\n")).toHaveLength(2);
  });

  it("formats VQA rows as '<video_id>, <frame_idx>, \"<answer>\"'", () => {
    const csv = buildSubmissionCsv({ task: "vqa", items: [item()], vqaAnswer: "5" });
    expect(csv).toBe('L01_V001, 750, "5"');
  });

  it("escapes double quotes inside the VQA answer", () => {
    const csv = buildSubmissionCsv({ task: "vqa", items: [item()], vqaAnswer: 'He said "hi"' });
    expect(csv).toBe('L01_V001, 750, "He said ""hi"""');
  });

  it("trims whitespace from the VQA answer before validating/embedding", () => {
    const csv = buildSubmissionCsv({ task: "vqa", items: [item()], vqaAnswer: "  Năm người  " });
    expect(csv).toBe('L01_V001, 750, "Năm người"');
  });

  it("rejects VQA export with an empty answer", () => {
    expect(() => buildSubmissionCsv({ task: "vqa", items: [item()], vqaAnswer: "" })).toThrow(
      ExportValidationError
    );
    expect(() => buildSubmissionCsv({ task: "vqa", items: [item()], vqaAnswer: "   " })).toThrow(
      ExportValidationError
    );
  });

  it("rejects a VQA answer longer than 100 characters", () => {
    const longAnswer = "a".repeat(101);
    expect(() => buildSubmissionCsv({ task: "vqa", items: [item()], vqaAnswer: longAnswer })).toThrow(
      ExportValidationError
    );
  });

  it("rejects an empty selection", () => {
    expect(() => buildSubmissionCsv({ task: "kis", items: [] })).toThrow(ExportValidationError);
  });

  it("rejects more than 100 rows instead of silently truncating", () => {
    const items = Array.from({ length: MAX_SUBMISSION_ROWS + 1 }, (_, i) =>
      item({ scene_id: `L01_V001_S${i}`, video_id: "L01_V001" })
    );
    expect(() => buildSubmissionCsv({ task: "kis", items })).toThrow(ExportValidationError);
  });

  it("accepts exactly 100 rows", () => {
    const items = Array.from({ length: MAX_SUBMISSION_ROWS }, (_, i) =>
      item({ scene_id: `L01_V001_S${i}`, video_id: "L01_V001" })
    );
    const csv = buildSubmissionCsv({ task: "kis", items });
    expect(csv.split("\n")).toHaveLength(MAX_SUBMISSION_ROWS);
  });

  it("outputs an empty frame_idx field when best_keyframe_id doesn't match the pattern", () => {
    const csv = buildSubmissionCsv({ task: "kis", items: [item({ best_keyframe_id: null })] });
    expect(csv).toBe("L01_V001, ");
  });
});
