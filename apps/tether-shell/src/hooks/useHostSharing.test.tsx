import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  listPeers,
  revokePeer,
  startHost,
  startPairing,
  stopHost,
} from "../ipc";
import type { Confirm } from "../state";
import { useHostSharing } from "./useHostSharing";

vi.mock("../ipc", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../ipc")>();
  return {
    ...actual,
    listPeers: vi.fn(() => Promise.resolve()),
    revokePeer: vi.fn(() => Promise.resolve()),
    startHost: vi.fn(() => Promise.resolve()),
    startPairing: vi.fn(() => Promise.resolve()),
    stopHost: vi.fn(() => Promise.resolve()),
  };
});

describe("useHostSharing", () => {
  beforeEach(() => {
    vi.mocked(listPeers).mockResolvedValue();
    vi.mocked(revokePeer).mockResolvedValue();
    vi.mocked(startHost).mockResolvedValue();
    vi.mocked(startPairing).mockResolvedValue();
    vi.mocked(stopHost).mockResolvedValue();
    vi.clearAllMocks();
  });

  it("tracks listening host state and refreshes paired peers", () => {
    const setConfirm = vi.fn();
    const { result } = renderHook(() => useHostSharing({ setConfirm }));

    act(() => {
      result.current.handleEvent({
        role: "host",
        event: "listening",
        addr: "192.168.1.20:7654",
        fingerprint: "host-fp",
      });
    });

    expect(result.current.host.running).toBe(true);
    expect(result.current.host.addr).toBe("192.168.1.20:7654");
    expect(result.current.host.error).toBeNull();
    expect(listPeers).toHaveBeenCalledTimes(1);
  });

  it("confirms before stopping sharing with a connected peer", () => {
    const setConfirm = vi.fn();
    const { result } = renderHook(() => useHostSharing({ setConfirm }));

    act(() => {
      result.current.handleEvent({
        role: "host",
        event: "peer_connected",
        peer: "192.168.1.30:51000",
      });
    });
    act(() => result.current.onStopSharing());

    const confirm = setConfirm.mock.calls[setConfirm.mock.calls.length - 1]?.[0] as Confirm;
    expect(confirm.title).toBe("Stop sharing?");
    expect(confirm.message).toBe("192.168.1.30:51000 is connected and will be disconnected.");

    act(() => confirm.onConfirm());

    expect(stopHost).toHaveBeenCalledTimes(1);
    expect(setConfirm).toHaveBeenCalledWith(null);
    expect(result.current.host.running).toBe(false);
    expect(result.current.host.peer).toBeNull();
  });

  it("requests pairing and revocation through typed IPC wrappers", () => {
    const setConfirm = vi.fn();
    const { result } = renderHook(() => useHostSharing({ setConfirm }));

    act(() => result.current.onAddDevice("Office laptop"));
    expect(startPairing).toHaveBeenCalledWith("Office laptop");

    act(() =>
      result.current.onRevoke({
        fingerprint: "peer-fp",
        label: "Office laptop",
        paired_at_unix: 1_700_000_000,
      }),
    );
    const confirm = setConfirm.mock.calls[setConfirm.mock.calls.length - 1]?.[0] as Confirm;

    act(() => confirm.onConfirm());

    expect(revokePeer).toHaveBeenCalledWith("peer-fp");
    expect(setConfirm).toHaveBeenCalledWith(null);
  });
});
