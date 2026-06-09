import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  getLoginStartEnabled,
  setLoginStartEnabled,
} from "../ipc";
import { initialHostState } from "../state";
import { SharingSheet } from "./SharingSheet";

vi.mock("../ipc", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../ipc")>();
  return {
    ...actual,
    getLoginStartEnabled: vi.fn(() => Promise.resolve(false)),
    setLoginStartEnabled: vi.fn(() => Promise.resolve(true)),
  };
});

function renderSheet() {
  return render(
    <SharingSheet
      host={initialHostState}
      onStart={vi.fn()}
      onStop={vi.fn()}
      onAddDevice={vi.fn()}
      onRevoke={vi.fn()}
      onCopy={vi.fn()}
      onClose={vi.fn()}
    />,
  );
}

describe("SharingSheet", () => {
  beforeEach(() => {
    vi.mocked(getLoginStartEnabled).mockResolvedValue(false);
    vi.mocked(setLoginStartEnabled).mockResolvedValue(true);
    vi.clearAllMocks();
  });

  it("loads and displays the login-start state", async () => {
    vi.mocked(getLoginStartEnabled).mockResolvedValue(true);

    renderSheet();

    const toggle = screen.getByRole("switch", { name: "Start Tether at login" });
    await waitFor(() => expect(toggle).toHaveAttribute("aria-checked", "true"));
    expect(getLoginStartEnabled).toHaveBeenCalledTimes(1);
  });

  it("toggles login-start without toggling sharing", async () => {
    const user = userEvent.setup();
    const onStart = vi.fn();
    const onStop = vi.fn();
    render(
      <SharingSheet
        host={initialHostState}
        onStart={onStart}
        onStop={onStop}
        onAddDevice={vi.fn()}
        onRevoke={vi.fn()}
        onCopy={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    const toggle = screen.getByRole("switch", { name: "Start Tether at login" });
    await waitFor(() => expect(toggle).not.toBeDisabled());
    await user.click(toggle);

    expect(setLoginStartEnabled).toHaveBeenCalledWith(true);
    expect(onStart).not.toHaveBeenCalled();
    expect(onStop).not.toHaveBeenCalled();
    await waitFor(() => expect(toggle).toHaveAttribute("aria-checked", "true"));
  });

  it("surfaces login-start errors inline", async () => {
    const user = userEvent.setup();
    vi.mocked(setLoginStartEnabled).mockRejectedValue(new Error("autostart denied"));
    renderSheet();

    const toggle = screen.getByRole("switch", { name: "Start Tether at login" });
    await waitFor(() => expect(toggle).not.toBeDisabled());
    await user.click(toggle);

    expect(await screen.findByText("Error: autostart denied")).toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-checked", "false");
  });
});
