//! Fakes for integration tests in downstream crates.
//!
//! Gated behind the `test-support` Cargo feature. Production builds
//! never compile this module.
//!
//! **Anti-drift rule**: `FakeDecoder` impls the production
//! `tether_codec::Decoder` trait. Refactors of the trait force a
//! compile error here before any downstream test runs.

#![deny(unused_imports)]

use std::collections::VecDeque;

use tether_codec::{CodecError, DecodedFrame, Decoder, Frame};
use tether_protocol::control::CodecKind;

/// Per-submit scripted outcome. Tests push these in the order they
/// want decode jobs to be handled.
pub enum FakeOutcome {
    /// `submit` succeeds, `next_frame` drains the supplied frames in
    /// order then returns `Ok(None)`.
    Frames(Vec<Frame>),
    /// `submit` returns an `Err` — the "hard failure" branch in
    /// `Worker::process_job`.
    SubmitError,
    /// `submit` succeeds, but `next_frame` returns an `Err` on the
    /// first pull. Tests the partial-drain failure branch.
    NextFrameError,
    /// Trivial single solid-color CPU frame at `(w, h)`. Convenience
    /// constructor — tests rarely care about pixel data.
    Solid {
        width: u32,
        height: u32,
        luma: u8,
    },
}

/// Decoder impl driven by a scripted queue of [`FakeOutcome`]s.
/// Exhausting the queue causes subsequent submits to return
/// `Ok(())` with zero frames, mirroring a quiescent decoder.
pub struct FakeDecoder {
    queue: VecDeque<FakeOutcome>,
    pending_frames: VecDeque<Frame>,
    next_frame_returns_error: bool,
}

impl FakeDecoder {
    pub fn new(queue: Vec<FakeOutcome>) -> Self {
        Self {
            queue: queue.into(),
            pending_frames: VecDeque::new(),
            next_frame_returns_error: false,
        }
    }

    /// Convenience: a decoder that succeeds every submit and emits one
    /// trivial frame per submit. Useful as a baseline "happy path".
    pub fn always_one_frame(width: u32, height: u32) -> Self {
        Self {
            queue: VecDeque::new(),
            pending_frames: VecDeque::new(),
            next_frame_returns_error: false,
        }
        .with_default_solid(width, height)
    }

    fn with_default_solid(mut self, width: u32, height: u32) -> Self {
        // Sentinel value triggers the default-solid path in submit.
        self.queue.push_back(FakeOutcome::Solid {
            width,
            height,
            luma: 0,
        });
        self
    }
}

impl Decoder for FakeDecoder {
    fn submit(&mut self, _encoded: &[u8]) -> Result<(), CodecError> {
        match self.queue.pop_front() {
            Some(FakeOutcome::Frames(fs)) => {
                self.pending_frames.extend(fs);
                Ok(())
            }
            Some(FakeOutcome::Solid {
                width,
                height,
                luma,
            }) => {
                self.pending_frames.push_back(solid_cpu_frame(width, height, luma));
                Ok(())
            }
            Some(FakeOutcome::SubmitError) => Err(CodecError::UnsupportedInputFormat),
            Some(FakeOutcome::NextFrameError) => {
                self.next_frame_returns_error = true;
                Ok(())
            }
            None => Ok(()),
        }
    }

    fn next_frame(&mut self) -> Result<Option<Frame>, CodecError> {
        if self.next_frame_returns_error {
            self.next_frame_returns_error = false;
            return Err(CodecError::UnsupportedInputFormat);
        }
        Ok(self.pending_frames.pop_front())
    }

    fn codec_kind(&self) -> CodecKind {
        CodecKind::H264
    }

    fn name(&self) -> &'static str {
        "FakeDecoder"
    }
}

fn solid_cpu_frame(width: u32, height: u32, luma: u8) -> Frame {
    let y_len = (width as usize) * (height as usize);
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    let uv_len = cw * ch * 2;
    Frame::Cpu(DecodedFrame {
        width,
        height,
        pts: None,
        y: vec![luma; y_len],
        uv: vec![0x80; uv_len],
    })
}
