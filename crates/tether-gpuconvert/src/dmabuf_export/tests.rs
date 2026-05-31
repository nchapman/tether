//! Integration tests for the single-plane export path. The shared NV12
//! export is exercised transitively through tether-gpuconvert's higher
//! BGRA→NV12 bridge tests (see `nv12_dmabuf.rs`).

use super::*;
use std::os::fd::AsRawFd;

async fn make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .ok()?;
    if !adapter
        .features()
        .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    {
        return None;
    }
    let required_features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("dmabuf export test"),
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .ok()
}

/// Plain export sanity. Doesn't try to use the DMA-BUF — just
/// validates the call returns a plausible descriptor (fd > 0,
/// stride >= row size, modifier as requested).
#[test]
#[ignore = "requires VK_EXT_external_memory_dma_buf (run on a real Linux GPU adapter)"]
fn export_r8unorm_smoke() {
    let Some((device, _queue)) = pollster::block_on(make_device()) else {
        eprintln!("SKIP: no wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF");
        return;
    };

    let width = 1920u32;
    let height = 1080u32;
    let export = export_texture_as_dmabuf(
        &device,
        width,
        height,
        wgpu::TextureFormat::R8Unorm,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        "test r8 y plane",
    )
    .expect("export r8");

    assert!(export.fd.as_raw_fd() >= 0, "fd must be valid");
    assert_eq!(export.drm_format_modifier, DRM_FORMAT_MOD_LINEAR);
    assert!(
        export.stride >= u64::from(width),
        "stride {} must be >= row width {}",
        export.stride,
        width
    );
    // Allocation must hold at least height * stride bytes.
    assert!(
        export.size >= u64::from(height) * export.stride,
        "size {} must be >= height({}) * stride({})",
        export.size,
        height,
        export.stride
    );
    // Offset is typically 0 for single-plane formats but the spec
    // allows non-zero. Just bound it.
    assert!(export.offset < export.size);
}

/// Round-trip: export → write a known pattern via wgpu →
/// re-import the same DMA-BUF as a separate texture →
/// copy_texture_to_buffer + readback → verify bytes match.
///
/// Proves the exported memory is *the same memory* the importer
/// sees (i.e. the fd actually references the right
/// VkDeviceMemory), and that the stride/offset/modifier values we
/// report let the importer access the data correctly. This is
/// the smallest test that validates the end-to-end export
/// contract without requiring VAAPI.
#[test]
#[ignore = "requires VK_EXT_external_memory_dma_buf"]
fn export_then_reimport_roundtrip() {
    let Some((device, queue)) = pollster::block_on(make_device()) else {
        eprintln!("SKIP: no wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF");
        return;
    };

    let width = 64u32;
    let height = 32u32;

    // Export with COPY_DST so we can write via queue.write_texture,
    // plus COPY_SRC isn't needed on the export side — the importer
    // is the one reading. STORAGE_BINDING is irrelevant for this
    // test but present in the production usage path.
    let export = export_texture_as_dmabuf(
        &device,
        width,
        height,
        wgpu::TextureFormat::R8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
        "roundtrip export",
    )
    .expect("export");

    // Write a recognisable pattern. With LINEAR tiling, byte
    // (x, y) at offset `y * stride + x` should be `(x ^ y) as u8`
    // — distinguishable even at small sizes.
    let mut bytes = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            bytes[(y * width + x) as usize] = ((x ^ y) & 0xff) as u8;
        }
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &export.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    // Submit + wait so the write lands in memory before the
    // import reads it. wgpu's write_texture is queued; we need
    // the queue to drain.
    queue.submit(std::iter::empty());
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll write");

    // Re-import via wgpu's existing import path. `try_clone` dups
    // the fd because texture_from_dmabuf_fd takes ownership.
    let import_fd = export.fd.try_clone().expect("dup fd for re-import");
    let import_desc = wgpu::hal::TextureDescriptor {
        label: Some("roundtrip import"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };
    // SAFETY: import_fd was just produced by us via the export
    // path, so the modifier/stride/offset we pass match.
    let import_hal = unsafe {
        device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .expect("vulkan backend")
            .texture_from_dmabuf_fd(
                import_fd,
                &import_desc,
                export.drm_format_modifier,
                export.stride,
                export.offset,
            )
            .expect("texture_from_dmabuf_fd")
    };
    let import_wgpu_desc = wgpu::TextureDescriptor {
        label: Some("roundtrip import"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    // SAFETY: hal texture built from this device on the same
    // backend; descs match shape.
    let import_tex = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(import_hal, &import_wgpu_desc)
    };

    // Copy the imported texture into a readback buffer and verify.
    let padded_row = width.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(padded_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback enc"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &import_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll readback");
    rx.recv().expect("map callback").expect("map");
    let mapped = slice.get_mapped_range().expect("get_mapped_range");

    for y in 0..height {
        for x in 0..width {
            let got = mapped[(y * padded_row + x) as usize];
            let want = ((x ^ y) & 0xff) as u8;
            assert_eq!(
                got, want,
                "byte ({x},{y}): exported pattern didn't survive re-import"
            );
        }
    }
}

/// Same round-trip but at Rg8Unorm (NV12 UV plane format) and
/// with an *odd* width to exercise stride padding. The export
/// path's `vkGetImageSubresourceLayout` should report a stride
/// that the importer's per-row math respects.
#[test]
#[ignore = "requires VK_EXT_external_memory_dma_buf"]
fn export_rg8unorm_odd_width_roundtrip() {
    let Some((device, queue)) = pollster::block_on(make_device()) else {
        eprintln!("SKIP: no wgpu adapter with VULKAN_EXTERNAL_MEMORY_DMA_BUF");
        return;
    };

    let width = 63u32; // odd — forces driver to pad the row
    let height = 32u32;

    let export = export_texture_as_dmabuf(
        &device,
        width,
        height,
        wgpu::TextureFormat::Rg8Unorm,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
        "rg8 roundtrip",
    )
    .expect("export rg8");

    // Each texel is 2 bytes (R + G). Write a per-texel pattern:
    // R = x, G = y. Tight rows on the CPU side; wgpu's
    // write_texture handles any per-row padding internally.
    let row_bytes = (width * 2) as usize;
    let mut bytes = vec![0u8; row_bytes * height as usize];
    for y in 0..height {
        for x in 0..width {
            let i = (y as usize) * row_bytes + (x as usize) * 2;
            bytes[i] = (x & 0xff) as u8;
            bytes[i + 1] = (y & 0xff) as u8;
        }
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &export.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 2),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::empty());
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll write");

    // Re-import using the export's reported stride/offset/modifier.
    // Dup the fd then drop the export's own fd *before* the
    // import reads, to prove the fd handle's lifetime is
    // independent of the memory's lifetime (memory stays alive
    // via the export's wgpu::Texture even after the fd is gone).
    let import_fd = export.fd.try_clone().expect("dup");
    let original_fd_value = {
        use std::os::fd::AsRawFd;
        export.fd.as_raw_fd()
    };
    drop(export.fd);
    let _ = original_fd_value; // future-proofing if we ever assert on it

    let import_desc = wgpu::hal::TextureDescriptor {
        label: Some("rg8 import"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg8Unorm,
        usage: wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };
    // SAFETY: import_fd was just produced by the export path so
    // modifier/stride/offset are authoritative.
    let import_hal = unsafe {
        device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .expect("vulkan backend")
            .texture_from_dmabuf_fd(
                import_fd,
                &import_desc,
                export.drm_format_modifier,
                export.stride,
                export.offset,
            )
            .expect("texture_from_dmabuf_fd Rg8")
    };
    let import_wgpu_desc = wgpu::TextureDescriptor {
        label: Some("rg8 import"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    let import_tex = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(import_hal, &import_wgpu_desc)
    };

    // Read back. Rg8 is 2 bytes per texel.
    let row_pad = (width * 2).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rg8 readback"),
        size: u64::from(row_pad) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rg8 enc"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &import_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_pad),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("cb").expect("map");
    let mapped = slice.get_mapped_range().expect("range");

    for y in 0..height {
        for x in 0..width {
            let row_off = y * row_pad;
            let i = (row_off + x * 2) as usize;
            assert_eq!(
                mapped[i],
                (x & 0xff) as u8,
                "R[{x},{y}] mismatched (stride={})",
                export.stride
            );
            assert_eq!(mapped[i + 1], (y & 0xff) as u8, "G[{x},{y}] mismatched");
        }
    }
}
