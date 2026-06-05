//! GPU device identity for multi-GPU pinning (GitHub issue #16).
//!
//! On a multi-GPU host the wgpu/Vulkan dma-buf *producer* (this crate's
//! BGRA→YUV bridges) and the CUDA *importer* (NVENC encode / NVDEC decode)
//! must live on the same physical GPU, or the EGL→CUDA import hands the codec
//! device pointers from the wrong GPU and the copy faults.
//!
//! wgpu leads. The bridge picks its adapter (wgpu's `HighPerformance`
//! heuristic today) and the codec side *follows* by mapping the adapter's
//! 16-byte Vulkan `deviceUUID` to the matching CUDA ordinal
//! (`tether_codec::nvenc::cuda_ordinal_for_uuid`) and the matching EGL device.
//! NVIDIA guarantees `VkPhysicalDeviceIDProperties::deviceUUID` equals the CUDA
//! `cuDeviceGetUuid` value for one physical GPU, so the UUID is the cross-API
//! correlation key. [`device_uuid`] reads it off a live producer device.

use ash::vk;
use wgpu::hal::api::Vulkan;

/// The 16-byte Vulkan `VkPhysicalDeviceIDProperties::deviceUUID` of a live
/// wgpu device, or `None` when the device isn't Vulkan-backed (no dma-buf
/// export path there anyway).
///
/// This is the producer's GPU identity. The host reads it off the bridge it
/// will feed and hands it to the codec side so NVENC/NVDEC's CUDA context and
/// the EGL importer bind to the same physical GPU the dma-buf was allocated on.
#[must_use]
pub fn device_uuid(device: &wgpu::Device) -> Option<[u8; 16]> {
    // SAFETY: hal escape hatch — the raw instance / physical-device handles
    // are valid for the lifetime of the `hal::Device` borrow, and we only
    // read properties through them (no resource creation or destruction).
    unsafe {
        let hal = device.as_hal::<Vulkan>()?;
        let instance = hal.shared_instance().raw_instance();
        let physical = hal.raw_physical_device();
        Some(physical_device_uuid(instance, physical))
    }
}

/// Query `VkPhysicalDeviceIDProperties::deviceUUID` (core Vulkan 1.1, always
/// present on the 1.1+ instances wgpu's Vulkan backend creates).
///
/// # Safety
/// `instance` must be a live `ash::Instance` (≥ Vulkan 1.1) and `physical`
/// one of the physical devices it enumerated.
unsafe fn physical_device_uuid(instance: &ash::Instance, physical: vk::PhysicalDevice) -> [u8; 16] {
    let mut id_props = vk::PhysicalDeviceIDProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id_props);
    // SAFETY: `physical` belongs to `instance`; `props2` is a well-formed
    // properties2 chain with the ID-properties struct linked in via push_next.
    unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
    id_props.device_uuid
}
