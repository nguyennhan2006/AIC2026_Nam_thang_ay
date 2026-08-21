// Hai thao tác của người rà bài QA, kiểm bằng chính DOM mà họ bấm:
//   - sửa MỘT câu trả lời cho cả loạt dòng đã tick;
//   - gõ hạng để CHÈN một dòng vào vị trí đó (đẩy phần còn lại xuống).
//
// Kiểm ở mức component chứ không tách hàm thuần ra test riêng: chỗ dễ sai là
// phần ghép (tick bám theo `key` trong khi bảng liên tục đổi thứ tự, và nháp
// hạng bị chốt hai lần khi Enter rồi rời ô), không phải phép biến đổi mảng.

import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SubmissionBoard } from "./SubmissionBoard";
import type { TunerRow } from "./FrameTuner";
import type { QaResultItem } from "../types";

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
    scene_id: `L01_V00${rank}#0`,
    evidence_ids: [],
  };
}

const QA = [
  qaItem(1, 100, "ba người"),
  qaItem(2, 200, "3 người"),
  qaItem(3, 300, "3 nguoi"),
  qaItem(4, 400, "bốn người"),
];

function renderQaBoard(onEditRows?: (rows: TunerRow[]) => void) {
  render(
    <SubmissionBoard
      apiConfig={apiConfig} task="QA" kis={[]} qa={QA} trake={[]} avs={[]}
      onEditRows={onEditRows}
    />
  );
}

/** Thứ tự video đang nộp, đọc thẳng từ bảng. */
function order(): (string | undefined)[] {
  return screen.getAllByRole("listitem").map((row) => row.textContent?.match(/L01_V\d+/)?.[0]);
}

function rowText(index: number): string {
  return screen.getAllByRole("listitem")[index].textContent ?? "";
}

function rankInput(index: number): HTMLInputElement {
  return within(screen.getAllByRole("listitem")[index]).getByRole("spinbutton") as HTMLInputElement;
}

function tickBox(index: number): HTMLInputElement {
  return within(screen.getAllByRole("listitem")[index]).getByRole("checkbox") as HTMLInputElement;
}

describe("SubmissionBoard — sửa đáp án hàng loạt", () => {
  it("áp một câu trả lời cho đúng các dòng đã tick, chừa các dòng còn lại", () => {
    renderQaBoard();

    fireEvent.click(tickBox(0));
    fireEvent.click(tickBox(2));
    fireEvent.change(screen.getByLabelText("Câu trả lời áp cho các dòng đã tick"), {
      target: { value: "3 người" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Áp cho 2 dòng/ }));

    expect(rowText(0)).toContain("3 người");
    expect(rowText(1)).toContain("3 người"); // vốn đã đúng, không tick nên giữ nguyên
    expect(rowText(2)).toContain("3 người");
    expect(rowText(3)).toContain("bốn người");
    // Chỉ hai dòng ĐƯỢC TICK mới tính là sửa tay; dòng 2 trùng chữ là của model.
    expect(screen.getByText("2 đáp án sửa tay")).toBeInTheDocument();
    expect(screen.getAllByTitle("Đáp án đã sửa tay, khác câu của model")).toHaveLength(2);
  });

  it("shift-click tick cả dải giữa hai dòng", () => {
    renderQaBoard();

    fireEvent.click(tickBox(0));
    fireEvent.click(tickBox(3), { shiftKey: true });

    expect(screen.getByText("4/4")).toBeInTheDocument();
  });

  it("tick bám theo dòng chứ không theo vị trí khi bảng đổi thứ tự", () => {
    renderQaBoard();

    fireEvent.click(tickBox(3)); // V004
    const input = rankInput(3);
    fireEvent.change(input, { target: { value: "1" } });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(order()[0]).toBe("L01_V004");
    expect(tickBox(0).checked).toBe(true);
    expect(tickBox(1).checked).toBe(false);
  });

  it("khoá nút áp dụng khi chưa tick dòng nào", () => {
    renderQaBoard();
    expect(screen.getByRole("button", { name: /Áp cho 0 dòng/ })).toBeDisabled();
  });
});

describe("SubmissionBoard — xếp hạng bằng cách gõ số", () => {
  it("chèn dòng vào hạng đã gõ và đẩy phần còn lại xuống, không hoán đổi", () => {
    renderQaBoard();
    expect(order()).toEqual(["L01_V001", "L01_V002", "L01_V003", "L01_V004"]);

    const input = rankInput(3);
    fireEvent.change(input, { target: { value: "1" } });
    fireEvent.keyDown(input, { key: "Enter" });

    // Hoán đổi sẽ cho V004, V002, V003, V001. Chèn phải giữ nguyên thứ tự
    // tương đối của ba dòng bị đẩy xuống.
    expect(order()).toEqual(["L01_V004", "L01_V001", "L01_V002", "L01_V003"]);
  });

  it("Enter rồi rời ô chỉ dịch chuyển MỘT lần", () => {
    renderQaBoard();

    const input = rankInput(0);
    fireEvent.change(input, { target: { value: "3" } });
    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.blur(input);

    expect(order()).toEqual(["L01_V002", "L01_V003", "L01_V001", "L01_V004"]);
  });

  it("Escape huỷ nháp, không dịch chuyển gì", () => {
    renderQaBoard();

    const input = rankInput(0);
    fireEvent.change(input, { target: { value: "4" } });
    fireEvent.keyDown(input, { key: "Escape" });
    fireEvent.blur(input);

    expect(order()).toEqual(["L01_V001", "L01_V002", "L01_V003", "L01_V004"]);
  });

  it("kẹp hạng gõ quá tay về trong khoảng 1..N", () => {
    renderQaBoard();

    const input = rankInput(0);
    fireEvent.change(input, { target: { value: "99" } });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(order()).toEqual(["L01_V002", "L01_V003", "L01_V004", "L01_V001"]);
  });

  it("đưa cả khối đã tick tới một hạng, giữ thứ tự bên trong khối", () => {
    renderQaBoard();

    fireEvent.click(tickBox(2));
    fireEvent.click(tickBox(3));
    fireEvent.change(screen.getByLabelText("Hạng đích cho các dòng đã tick"), {
      target: { value: "1" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Chèn/ }));

    expect(order()).toEqual(["L01_V003", "L01_V004", "L01_V001", "L01_V002"]);
  });

  it("bỏ hàng loạt các dòng đã tick", () => {
    renderQaBoard();

    fireEvent.click(tickBox(0));
    fireEvent.click(tickBox(1));
    fireEvent.click(screen.getByRole("button", { name: /Bỏ 2 dòng đã tick/ }));

    expect(order()).toEqual(["L01_V003", "L01_V004"]);
    expect(screen.getByText("0/2")).toBeInTheDocument();
  });
});

// Toolbar cũ phải sống sót qua mọi lần thêm thao tác mới: "Lưu & chỉnh frame"
// là đường DUY NHẤT đưa đúng tập dòng (và thứ tự) của bảng nộp sang tab chỉnh
// frame — mất nó thì tab kia lại nạp từ kết quả tìm kiếm thô và người dùng
// chỉnh một danh sách khác với danh sách sắp nộp.
describe("SubmissionBoard — bàn giao sang tab chỉnh frame", () => {
  it("vẫn còn nút Lưu & chỉnh frame, và đẩy đúng thứ tự + đáp án đã sửa tay", () => {
    const onEditRows = vi.fn();
    renderQaBoard(onEditRows);

    // Sửa đáp án hàng loạt rồi đảo thứ tự, để chắc cả hai đều đi theo.
    fireEvent.click(tickBox(0));
    fireEvent.change(screen.getByLabelText("Câu trả lời áp cho các dòng đã tick"), {
      target: { value: "3 người" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Áp cho 1 dòng/ }));
    const input = rankInput(0);
    fireEvent.change(input, { target: { value: "2" } });
    fireEvent.keyDown(input, { key: "Enter" });

    fireEvent.click(screen.getByRole("button", { name: /Lưu & chỉnh frame/ }));

    const handed = onEditRows.mock.calls[0][0] as TunerRow[];
    expect(handed.map((row) => row.videoId)).toEqual([
      "L01_V002", "L01_V001", "L01_V003", "L01_V004",
    ]);
    expect(handed[1].answer).toBe("3 người");
  });
});
