import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// `globals: false` (mặc định của vite.config.ts) nên auto-cleanup của RTL
// không tự đăng ký — thiếu dòng này thì mỗi test render thêm một bản nữa vào
// cùng document, và getAllByRole của test sau đếm cả các bảng của test trước.
afterEach(cleanup);
