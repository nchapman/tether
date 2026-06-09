import { DEFAULT_PORT } from "./appChannel";

const PORT_RE = /^\d{1,5}$/;
const IPV4_PART_RE = /^\d{1,3}$/;
const BRACKETED_IPV6_SOCKET_RE = /^\[([0-9a-fA-F:.]+)\]:(\d{1,5})$/;
const IPV4_SOCKET_RE = /^(.+):(\d{1,5})$/;

export { DEFAULT_PORT };

// Append the default port unless the address already has one. Handles IPv6:
// a bare literal (`fe80::1`) is bracketed with the current default port so it
// parses as a SocketAddr, and an already-bracketed address keeps its port.
function withDefaultPort(addr: string, defaultPort: string): string {
  if (addr.startsWith("[")) {
    // Bracketed IPv6, with or without a trailing `:port`.
    return addr.includes("]:") ? addr : `${addr}:${defaultPort}`;
  }
  let colons = 0;
  for (const char of addr) {
    if (char === ":") colons += 1;
  }
  if (colons === 1) return addr; // host:port or IPv4:port
  if (colons > 1) return `[${addr}]:${defaultPort}`; // bare IPv6 literal
  return `${addr}:${defaultPort}`; // bare IPv4 or unsupported hostname
}

function validPort(raw: string): boolean {
  if (PORT_RE.exec(raw) === null) return false;
  const port = Number(raw);
  return port > 0 && port <= 65535;
}

function validIpv4(raw: string): boolean {
  const parts = raw.split(".");
  return (
    parts.length === 4 &&
    parts.every((part) => IPV4_PART_RE.exec(part) !== null && Number(part) <= 255)
  );
}

function validBracketedIpv6Socket(raw: string): boolean {
  const match = BRACKETED_IPV6_SOCKET_RE.exec(raw);
  return match !== null && match[1].includes(":") && validPort(match[2]);
}

export function normalizeSocketAddr(raw: string, defaultPort = DEFAULT_PORT): string | null {
  const candidate = withDefaultPort(raw, defaultPort);
  const ipv4 = IPV4_SOCKET_RE.exec(candidate);
  if (ipv4 !== null && validIpv4(ipv4[1]) && validPort(ipv4[2])) {
    return candidate;
  }
  if (validBracketedIpv6Socket(candidate)) {
    return candidate;
  }
  return null;
}
