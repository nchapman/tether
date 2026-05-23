//! Decoder worker for the client.
//!
//! Owns the libavcodec/VAAPI decoder and the auto-IDR rate-limit
//! state. Lives on a dedicated `std::thread` (not a tokio worker) so
//! that a GPU-driver stall — `vaSyncSurface`, `vaExportSurfaceHandle`,
//! decoder bind operations — can't starve the recv loop's QUIC
//! datagram polling.
//!
//! Two failure shapes trigger the same recovery path: a hard
//! `decoder.submit`/`next_frame` error, and a soft failure where
//! libavcodec internally logged at warning-or-above during decode
//! (the typical "P-frame references a fragment we dropped on the
//! wire" case where the codec silently skips a NALU). Either way the
//! only recovery is a fresh IDR, requested through the
//! [`Worker::request_idr`] callback and rate-limited by
//! [`IDR_RATE_LIMIT`].

pub mod run;

#[cfg(feature = "test-support")]
pub mod test_support;

pub use run::{
    run_thread, run_thread_with_init, DecodeCompletion, DecodeJob, RecoveryAction, Worker,
    IDR_RATE_LIMIT, NO_OUTPUT_WATCHDOG, REBUILD_BUDGET, TRANSIENT_FAILURE_THRESHOLD,
};
