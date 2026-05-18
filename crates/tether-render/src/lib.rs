//! Client-side display: a winit window driving a wgpu render pipeline,
//! fed from a crossbeam channel of [`RawFrame`]s.
//!
//! Task #4 (minimal): RGBA passthrough only. Task #9 will add a YUV→RGB
//! fragment shader, an adaptive jitter buffer, zero-copy decoded-texture
//! import, and present-time telemetry hooks.

mod gpu;

use std::sync::Arc;

use crossbeam_channel::Receiver;
use tracing::warn;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use gpu::GpuState;

/// A single uncompressed RGBA frame. Pixel layout is row-major,
/// `width * height * 4` bytes total, R first, alpha last.
#[derive(Clone, Debug)]
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("event loop: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error("winit OS: {0}")]
    Os(#[from] winit::error::OsError),
    #[error("no compatible wgpu adapter")]
    NoAdapter,
    #[error("wgpu request device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("wgpu surface create: {0}")]
    SurfaceCreate(#[from] wgpu::CreateSurfaceError),
}

pub type Result<T> = std::result::Result<T, RenderError>;

/// Run the render loop on the current thread. Blocks until the user closes
/// the window. The frame channel may disconnect; the window stays open
/// showing the last received frame until the user closes it.
pub fn run(
    title: &str,
    initial_size: (u32, u32),
    frames: Receiver<RawFrame>,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App {
        title: title.to_string(),
        initial_size,
        window: None,
        gpu: None,
        frames,
        latest: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    title: String,
    initial_size: (u32, u32),
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    frames: Receiver<RawFrame>,
    latest: Option<RawFrame>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title(&self.title)
            .with_inner_size(LogicalSize::new(self.initial_size.0, self.initial_size.1));
        let win = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                tracing::error!(error = %e, "failed to create window");
                event_loop.exit();
                return;
            }
        };
        let gpu = match pollster::block_on(GpuState::new(win.clone())) {
            Ok(g) => g,
            Err(e) => {
                tracing::error!(error = %e, "failed to initialise wgpu");
                event_loop.exit();
                return;
            }
        };
        self.window = Some(win);
        self.gpu = Some(gpu);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                if let Some(frame) = &self.latest {
                    gpu.upload(frame);
                }
                if let Err(e) = gpu.render() {
                    warn!(error = ?e, "render frame failed");
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let mut received = false;
        loop {
            match self.frames.try_recv() {
                Ok(frame) => {
                    self.latest = Some(frame);
                    received = true;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Producer is gone — keep showing the last frame
                    // until the user closes the window.
                    break;
                }
            }
        }
        if received {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}
