import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AddComputerSheet } from "./AddComputerSheet";

async function enterPin(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByLabelText("Digit 1"));
  await user.keyboard("12345678");
}

describe("AddComputerSheet", () => {
  it("normalizes bare IPv4 addresses with the default port", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <AddComputerSheet
        status="idle"
        onSubmit={onSubmit}
        onClose={vi.fn()}
      />,
    );

    await user.type(screen.getByPlaceholderText("192.168.1.10"), "192.168.1.20");
    await enterPin(user);
    await user.click(screen.getByRole("button", { name: /pair/i }));

    expect(onSubmit).toHaveBeenCalledWith("192.168.1.20:7654", "12345678", "");
  });

  it("brackets bare IPv6 literals before submitting", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <AddComputerSheet
        prefillAddr="fe80::1"
        status="idle"
        onSubmit={onSubmit}
        onClose={vi.fn()}
      />,
    );

    await enterPin(user);
    await user.click(screen.getByRole("button", { name: /pair/i }));

    expect(onSubmit).toHaveBeenCalledWith("[fe80::1]:7654", "12345678", "");
  });

  it("does not submit invalid socket addresses", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <AddComputerSheet
        status="idle"
        onSubmit={onSubmit}
        onClose={vi.fn()}
      />,
    );

    await user.type(screen.getByPlaceholderText("192.168.1.10"), "999.1.1.1:7654");
    await enterPin(user);

    expect(screen.getByRole("button", { name: /pair/i })).toBeDisabled();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
