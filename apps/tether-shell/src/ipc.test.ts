import { describe, expect, it } from "vitest";

import { friendlyError, recencyLabel, type SavedHost } from "./ipc";

const host: SavedHost = {
  addr: "192.168.1.20:7654",
  label: "Office desktop",
  paired_at_unix: 1_700_000_000,
  last_connected_unix: null,
};

describe("friendlyError", () => {
  it("turns lost trust into a pair-again action", () => {
    expect(friendlyError("refused resume: unknown peer", "Office desktop")).toEqual({
      message:
        "Office desktop no longer recognizes this computer. You'll need to pair again with a new PIN.",
      pairAgain: true,
    });
  });

  it("turns pairing failures into user-actionable PIN guidance", () => {
    expect(friendlyError("pairing failed: pin expired", "Office desktop")).toEqual({
      message:
        "That PIN didn't match — it may be wrong or expired. Check the PIN on the other computer and try again.",
      pairAgain: false,
    });
  });

  it("keeps unknown errors intact", () => {
    expect(friendlyError("codec probe failed", "Office desktop")).toEqual({
      message: "codec probe failed",
      pairAgain: false,
    });
  });
});

describe("recencyLabel", () => {
  it("labels recent connects from last_connected_unix", () => {
    expect(
      recencyLabel(
        { ...host, last_connected_unix: 1_700_000_000 },
        1_700_000_000_000 + 2 * 60 * 60 * 1000,
      ),
    ).toBe("2h ago");
  });

  it("prefixes first-paired rows", () => {
    expect(recencyLabel(host, 1_700_000_000_000 + 24 * 60 * 60 * 1000)).toBe(
      "paired yesterday",
    );
  });
});
