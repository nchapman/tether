import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  connectClient,
  disconnectClient,
  forgetKnownHost,
  hideWindow,
  renameKnownHost,
  showWindow,
} from "../ipc";
import type { Confirm } from "../state";
import { useClientSession, type ClientSessionDeps } from "./useClientSession";

vi.mock("../ipc", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../ipc")>();
  return {
    ...actual,
    connectClient: vi.fn(() => Promise.resolve()),
    disconnectClient: vi.fn(() => Promise.resolve()),
    forgetKnownHost: vi.fn(() => Promise.resolve()),
    hideWindow: vi.fn(() => Promise.resolve()),
    renameKnownHost: vi.fn(() => Promise.resolve()),
    showWindow: vi.fn(() => Promise.resolve()),
  };
});

function renderClientSession(overrides: Partial<ClientSessionDeps> = {}) {
  const windowHidden = { current: false };
  const deps: ClientSessionDeps = {
    windowHiddenRef: windowHidden,
    restoreWindow: vi.fn(() => {
      windowHidden.current = false;
    }),
    refreshHosts: vi.fn(),
    labelFor: (addr) =>
      ({
        "192.168.1.20:7654": "Office desktop",
        "192.168.1.30:7654": "Laptop",
      })[addr] ?? addr,
    setSheet: vi.fn(),
    setConfirm: vi.fn(),
    ...overrides,
  };

  return {
    windowHidden,
    deps,
    hook: renderHook(() => useClientSession(deps)),
  };
}

describe("useClientSession", () => {
  beforeEach(() => {
    vi.mocked(connectClient).mockResolvedValue();
    vi.mocked(disconnectClient).mockResolvedValue();
    vi.mocked(forgetKnownHost).mockResolvedValue();
    vi.mocked(hideWindow).mockResolvedValue();
    vi.mocked(renameKnownHost).mockResolvedValue();
    vi.mocked(showWindow).mockResolvedValue();
    vi.clearAllMocks();
  });

  it("moves from connecting to connected and hides the shell window", async () => {
    const { deps, hook, windowHidden } = renderClientSession();

    await act(async () => {
      await hook.result.current.beginConnect("192.168.1.20:7654", "saved");
    });
    expect(hook.result.current.client).toEqual({
      kind: "connecting",
      addr: "192.168.1.20:7654",
      via: "saved",
    });

    act(() => {
      hook.result.current.handleEvent({
        role: "client",
        event: "connected",
        host: "192.168.1.20:7654",
        profile: "HEVC Main",
      });
    });

    expect(hook.result.current.client).toEqual({
      kind: "connected",
      addr: "192.168.1.20:7654",
      profile: "HEVC Main",
    });
    expect(deps.refreshHosts).toHaveBeenCalledTimes(1);
    expect(deps.setSheet).toHaveBeenCalledWith("none");
    expect(hideWindow).toHaveBeenCalledTimes(1);
    expect(windowHidden.current).toBe(true);
  });

  it("surfaces saved-row trust failures as pair-again row errors", async () => {
    vi.mocked(connectClient).mockRejectedValueOnce(
      new Error("connect failed: pairing required"),
    );
    const { hook } = renderClientSession();

    await act(async () => {
      await hook.result.current.beginConnect("192.168.1.20:7654", "saved");
    });

    expect(hook.result.current.client).toEqual({ kind: "idle" });
    expect(hook.result.current.rowErrors["192.168.1.20:7654"]).toEqual({
      message:
        "Office desktop no longer recognizes this computer. You'll need to pair again with a new PIN.",
      pairAgain: true,
    });
  });

  it("confirms before switching away from a live session", async () => {
    const setConfirm = vi.fn();
    const { hook } = renderClientSession({ setConfirm });

    act(() => {
      hook.result.current.handleEvent({
        role: "client",
        event: "connected",
        host: "192.168.1.20:7654",
        profile: "HEVC Main",
      });
    });
    act(() => hook.result.current.onConnect("192.168.1.30:7654"));

    const confirm = setConfirm.mock.calls[setConfirm.mock.calls.length - 1]?.[0] as Confirm;
    expect(confirm.title).toBe("Switch computer?");
    expect(confirm.message).toBe("Disconnect from Office desktop and connect to Laptop?");

    act(() => confirm.onConfirm());

    await waitFor(() =>
      expect(connectClient).toHaveBeenCalledWith({
        addr: "192.168.1.30:7654",
        pin: null,
        label: null,
      }),
    );
    expect(setConfirm).toHaveBeenCalledWith(null);
  });
});
