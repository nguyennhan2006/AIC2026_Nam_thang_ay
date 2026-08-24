// FB-003: bảng nộp và lưới ảnh phải nói CÙNG một thứ về đáp án, và bản nháp
// nạp vào phải nói thật về những dòng nó không khớp được.
//
// Hai chỗ dễ hỏng, cả hai đều hỏng IM LẶNG:
//   1. sửa đáp án hàng loạt xong, lưới ảnh vẫn hiện câu cũ của model;
//   2. nạp bản nháp của truy vấn KHÁC, dòng không khớp bị bỏ mà không ai báo —
//      người dùng build CSV rồi nộp thiếu dòng.

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { QaResultItem, SceneAnswer, SearchHit, SubmissionDraft } from "../types";

const listDrafts = vi.fn();
const saveDraft = vi.fn();
const deleteDraft = vi.fn();

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, listDrafts, saveDraft, deleteDraft };
});

const { SubmissionBoard } = await import("./SubmissionBoard");
const { ResultsExplorer } = await import("./ResultsExplorer");

const apiConfig = { base: "http://localhost:8000", token: "" };

function qaItem(rank: number, frameIdx: number, answer: string): QaResultItem {
  return {
    rank,
    video_id: `L01_V00${rank}`,
    frame_idx: frameIdx,
    answer,
    canonical_answer: answer.toLowerCase(),
    answer_type: "text",
    joint_score: 1 / rank,
    verifier_status: "SUPPORTED",
    scene_id: `L01_V00${rank}_S000`,
    evidence_ids: [],
  };
}

const QA = [qaItem(1, 100, "ba người"), qaItem(2, 200, "3 người"), qaItem(3, 300, "3 nguoi")];

function draft(rows: SubmissionDraft["rows"], name = "bản của Nhân"): SubmissionDraft {
  return {
    draft_id: "d1",
    name,
    author: "nhan",
    task: "QA",
    query: "có mấy người",
    rows,
    created_at: "2026-08-22T10:00:00+00:00",
    updated_at: "2026-08-22T10:00:00+00:00",
  };
}

function row(videoId: string, frameIdx: number, answer: string | null = null) {
  return { video_id: videoId, frame_idx: frameIdx, frame_ids: [], answer };
}

beforeEach(() => {
  listDrafts.mockReset().mockResolvedValue([]);
  saveDraft.mockReset();
  deleteDraft.mockReset();
});

function renderBoard(onAnswersChange?: (answers: SceneAnswer[]) => void) {
  render(
    <SubmissionBoard
      apiConfig={apiConfig} task="QA" kis={[]} qa={QA} trake={[]} avs={[]}
      query="có mấy người" onAnswersChange={onAnswersChange}
    />
  );
}

/** DraftBar gọi `listDrafts` trong effect lúc mount, nên danh sách phải được
 *  mock TRƯỚC khi render — đổi mock sau đó không kích hoạt lại lần nạp nào. */
async function renderWithDraft(item: SubmissionDraft) {
  listDrafts.mockResolvedValue([item]);
  renderBoard();
  await waitFor(() =>
    expect(screen.getByLabelText("Bản nháp của cả đội")).toHaveTextContent(item.name)
  );
  return item;
}

async function pickAndLoad(item: SubmissionDraft) {
  fireEvent.change(screen.getByLabelText("Bản nháp của cả đội"), {
    target: { value: item.draft_id },
  });
  fireEvent.click(screen.getByRole("button", { name: "Nạp" }));
}

describe("đáp án bảng nộp -> lưới ảnh", () => {
  it("công bố đáp án của model ngay khi có kết quả", async () => {
    const onAnswersChange = vi.fn();
    renderBoard(onAnswersChange);

    await waitFor(() => expect(onAnswersChange).toHaveBeenCalled());
    const latest = onAnswersChange.mock.calls.at(-1)![0] as SceneAnswer[];
    expect(latest.map((item) => item.answer)).toEqual(["ba người", "3 người", "3 nguoi"]);
    expect(latest.every((item) => !item.edited)).toBe(true);
    expect(latest[0].sceneId).toBe("L01_V001_S000");
  });

  it("sửa hàng loạt xong thì đáp án công bố ra ngoài đổi theo, kèm câu gốc", async () => {
    const onAnswersChange = vi.fn();
    renderBoard(onAnswersChange);

    const box = (index: number) =>
      within(screen.getAllByRole("listitem")[index]).getByRole("checkbox");
    fireEvent.click(box(0));
    fireEvent.click(box(2));
    fireEvent.change(screen.getByLabelText("Câu trả lời áp cho các dòng đã tick"), {
      target: { value: "3 người" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Áp cho 2 dòng/ }));

    const latest = onAnswersChange.mock.calls.at(-1)![0] as SceneAnswer[];
    expect(latest.map((item) => item.answer)).toEqual(["3 người", "3 người", "3 người"]);
    expect(latest.map((item) => item.edited)).toEqual([true, false, true]);
    // Câu của model phải còn nguyên để người soát biết mình vừa đè lên cái gì.
    expect(latest[0].modelAnswer).toBe("ba người");
  });

  it("lưới ảnh vẽ đúng đáp án đã sửa và đánh dấu là sửa tay", () => {
    const hit = {
      playback: null, rank: 1, candidate_id: "c1", scene_id: "L01_V001_S000",
      video_id: "L01_V001", video_path: null, event_id: null, scene_idx: 0,
      start_frame: 0, end_frame_exclusive: 300, start_sec: 0, end_sec: 10,
      best_frame_idx: 100, best_keyframe_id: null, best_keyframe_path: null,
      best_timestamp_sec: 3.3, safe_frame_score: null, score: 0.9, keyframes: [],
      matched_modalities: [], matched_branches: [], evidence: [],
      component_scores: {}, branch_contributions: {}, warnings: [],
    } as unknown as SearchHit;

    render(
      <ResultsExplorer
        results={[hit]} sequences={[]} apiConfig={apiConfig}
        selectedCandidateId={null} onSelect={() => {}} pristine={false}
        topK={20} perVideoCapSet={false}
        answerBySceneId={new Map([[
          "L01_V001_S000",
          {
            sceneId: "L01_V001_S000", videoId: "L01_V001", frameIdx: 100,
            answer: "3 người", modelAnswer: "ba người", edited: true,
          },
        ]])}
      />
    );

    expect(screen.getByText("3 người")).toBeInTheDocument();
    expect(screen.getByText("sửa")).toBeInTheDocument();
    expect(
      screen.getByTitle('Đáp án đã sửa tay ở bảng nộp. Câu của model: “ba người”')
    ).toBeInTheDocument();
  });
});

describe("nạp bản nháp", () => {
  it("áp đúng thứ tự đã lưu và đáp án đã sửa", async () => {
    await pickAndLoad(
      await renderWithDraft(
        draft([row("L01_V003", 300, "ba người"), row("L01_V001", 100), row("L01_V002", 200)])
      )
    );

    const order = screen.getAllByRole("listitem").map((item) => item.textContent?.match(/L01_V\d+/)?.[0]);
    expect(order).toEqual(["L01_V003", "L01_V001", "L01_V002"]);
    expect(screen.getAllByRole("listitem")[0].textContent).toContain("ba người");
    expect(screen.getByText(/Đã nạp "bản của Nhân": 3 dòng/)).toBeInTheDocument();
  });

  it("NÓI RA số dòng không khớp thay vì bỏ im lặng", async () => {
    await pickAndLoad(
      await renderWithDraft(
        draft([row("L01_V001", 100), row("L09_V999", 42), row("L08_V888", 7)], "nháp câu khác")
      )
    );

    // Bỏ im lặng là cách nhanh nhất để nộp thiếu dòng mà không ai biết.
    expect(screen.getByText(/1\/3 dòng khớp kết quả hiện tại/)).toBeInTheDocument();
    expect(screen.getByText(/2 dòng KHÔNG có trong kết quả này/)).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });

  it("cảnh báo khi bản nháp thuộc task khác", async () => {
    const item = await renderWithDraft({
      ...draft([row("L01_V001", 100)]),
      task: "TRAKE" as const,
    });
    fireEvent.change(screen.getByLabelText("Bản nháp của cả đội"), {
      target: { value: item.draft_id },
    });

    expect(screen.getByText(/thuộc task TRAKE, bạn đang ở QA/)).toBeInTheDocument();
  });
});

describe("lưu bản nháp", () => {
  it("gửi đúng thứ tự đang hiện và đáp án đã sửa", async () => {
    saveDraft.mockResolvedValue(draft([]));
    renderBoard();

    // Đảo thứ tự trước khi lưu, để chắc bản nháp chở thứ tự CỦA NGƯỜI DÙNG.
    const rankInput = within(screen.getAllByRole("listitem")[2]).getByRole("spinbutton");
    fireEvent.change(rankInput, { target: { value: "1" } });
    fireEvent.keyDown(rankInput, { key: "Enter" });

    fireEvent.change(screen.getByLabelText("Tên người lưu bản nháp"), { target: { value: "nhan" } });
    fireEvent.change(screen.getByLabelText("Tên bản nháp"), { target: { value: "câu 7" } });
    fireEvent.click(screen.getByRole("button", { name: /Lưu nháp/ }));

    await waitFor(() => expect(saveDraft).toHaveBeenCalled());
    const body = saveDraft.mock.calls[0][1];
    expect(body.name).toBe("câu 7");
    expect(body.author).toBe("nhan");
    expect(body.query).toBe("có mấy người");
    expect(body.rows.map((item: { video_id: string }) => item.video_id)).toEqual([
      "L01_V003", "L01_V001", "L01_V002",
    ]);
    expect(body.draft_id).toBeNull();
  });
});
