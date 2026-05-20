//! Cross-platform session helpers shared between host and client.
//!
//! This crate is the seam where macOS and Windows host/client binaries
//! will plug into the existing Linux orchestration without duplicating
//! the platform-agnostic pieces: IDR coalescing, encode/decode rolling
//! stats, and (incrementally) more of the per-connection loop.
//!
//! Current scope is intentionally narrow: only the helpers whose value
//! is obvious independent of a second platform actually existing live
//! here. As macOS and Windows host binaries land, the platform-shared
//! orchestration (handshake, fragmenter ownership, the recv-task
//! JoinSet) will move in here too.

mod idr;
mod stats;

pub use idr::IdrSignal;
pub use stats::EncodeStatsWindow;
