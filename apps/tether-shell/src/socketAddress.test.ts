import { describe, expect, it } from "vitest";

import { normalizeSocketAddr } from "./socketAddress";

describe("normalizeSocketAddr", () => {
  it("adds the default port to bare IPv4 addresses", () => {
    expect(normalizeSocketAddr("192.168.1.20", "7374")).toBe("192.168.1.20:7374");
    expect(normalizeSocketAddr("192.168.1.20", "7384")).toBe("192.168.1.20:7384");
  });

  it("preserves explicit IPv4 ports", () => {
    expect(normalizeSocketAddr("192.168.1.20:9000")).toBe("192.168.1.20:9000");
  });

  it("brackets bare IPv6 literals", () => {
    expect(normalizeSocketAddr("fe80::1", "7384")).toBe("[fe80::1]:7384");
  });

  it("rejects invalid addresses and ports", () => {
    expect(normalizeSocketAddr("999.1.1.1:7374")).toBeNull();
    expect(normalizeSocketAddr("192.168.1.20:70000")).toBeNull();
  });
});
