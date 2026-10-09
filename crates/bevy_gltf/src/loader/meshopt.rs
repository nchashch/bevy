//! [`EXT_meshopt_compression`](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_meshopt_compression/)
//! buffer decoding.
//!
//! `gltfpack -c` packs all mesh/vertex data into buffer views compressed with
//! the meshoptimizer codec. Those views point at a *virtual* buffer (declared
//! with `EXT_meshopt_compression: { fallback: true }` and no data) that holds
//! the decoded contents, while the extension points at the compressed range
//! inside a real buffer. This module decodes the compressed ranges into the
//! virtual buffers after the raw buffers are loaded, so accessors can read
//! them through the normal buffer path.
//!
//! Note: this crate forbids `unsafe`, so the meshoptimizer C decoder is
//! accessed through the safe API of the `meshopt` crate. That API does not
//! expose index-sequence decoding or the octahedral/quaternion/exponential
//! filters — views using those are rejected with an error (gltfpack's default
//! output does not use them).

use meshopt::encoding::{decode_index_buffer, decode_vertex_buffer};
use serde_json::Value;

use super::GltfError;

const MESHOPT_EXT: &str = "EXT_meshopt_compression";

/// Decode failure helper.
fn invalid(msg: &str) -> GltfError {
    GltfError::Gltf(gltf::Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        msg.to_string(),
    )))
}

#[derive(Debug)]
struct MeshoptView {
    /// Offset of the decoded data within the virtual buffer.
    byte_offset: u64,
    decoded_byte_length: u64,
    /// Buffer holding the compressed data + its range within it.
    source_buffer: usize,
    ext_byte_offset: u64,
    ext_byte_length: u64,
    mode: MeshoptMode,
    count: u64,
    stride: u64,
    filter: MeshoptFilter,
}

#[derive(Debug)]
enum MeshoptMode {
    Attributes,
    Triangles,
}

#[derive(Debug)]
enum MeshoptFilter {
    None,
}

#[derive(Debug)]
pub(super) struct MeshoptBufferPlan {
    /// The virtual decoded buffer this plan fills.
    buffer_index: usize,
    /// The buffer holding the compressed data.
    source_buffer: usize,
    views: Vec<MeshoptView>,
    /// Total size of the decoded buffer.
    decoded_len: u64,
}

/// Extracts the EXT_meshopt_compression description of every buffer view from
/// the raw glTF JSON. Returns one plan per affected (virtual) buffer, plus
/// the indices of virtual buffers that must be skipped by buffer loading.
pub(super) fn parse_meshopt_plans(
    bytes: &[u8],
) -> Result<(Vec<MeshoptBufferPlan>, Vec<usize>), GltfError> {
    // GLB containers hold the JSON in the first chunk; plain .gltf is JSON
    // throughout.
    let json_bytes: &[u8] = if bytes.len() >= 4 && &bytes[0..4] == b"glTF" {
        if bytes.len() < 20 {
            return Err(invalid("GLB container too small"));
        }
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        &bytes[20..20 + json_len]
    } else {
        bytes
    };

    let root: Value = serde_json::from_slice(json_bytes)
        .map_err(|err| invalid(&format!("Failed to parse glTF JSON: {err}")))?;

    let missing = |key: &str| invalid(&format!("{MESHOPT_EXT} is missing `{key}`"));

    let mut plans: Vec<MeshoptBufferPlan> = Vec::new();
    let mut meshopt_buffers = std::collections::HashSet::new();
    let mut plain_buffers = std::collections::HashSet::new();
    let mut virtual_buffers = Vec::new();
    for view in root
        .get("bufferViews")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let buffer_index = view
            .get("buffer")
            .and_then(Value::as_u64)
            .ok_or_else(|| missing("buffer"))? as usize;
        let Some(ext) = view.get("extensions").and_then(|e| e.get(MESHOPT_EXT)) else {
            if meshopt_buffers.contains(&buffer_index) {
                return Err(invalid(
                    "Buffers mixing meshopt-compressed and plain views are not supported",
                ));
            }
            plain_buffers.insert(buffer_index);
            continue;
        };
        if plain_buffers.contains(&buffer_index) {
            return Err(invalid(
                "Buffers mixing meshopt-compressed and plain views are not supported",
            ));
        }
        let plan = match plans.last_mut() {
            Some(plan) if plan.buffer_index == buffer_index => plan,
            _ => {
                plans.push(MeshoptBufferPlan {
                    buffer_index,
                    source_buffer: 0,
                    views: Vec::new(),
                    decoded_len: 0,
                });
                plans.last_mut().unwrap()
            }
        };
        meshopt_buffers.insert(buffer_index);
        let source_buffer = ext
            .get("buffer")
            .and_then(Value::as_u64)
            .ok_or_else(|| missing("buffer"))? as usize;
        plan.source_buffer = source_buffer;

        let mode = match ext.get("mode").and_then(Value::as_str) {
            Some("ATTRIBUTES") => MeshoptMode::Attributes,
            Some("TRIANGLES") => MeshoptMode::Triangles,
            Some(other) => {
                return Err(invalid(&format!("Unknown meshopt mode: {other}")));
            }
            None => return Err(missing("mode")),
        };
        let filter = match ext.get("filter").and_then(Value::as_str) {
            Some("NONE") | None => MeshoptFilter::None,
            Some(other) => {
                return Err(invalid(&format!(
                    "Unsupported meshopt filter: {other} (only NONE is supported)"
                )));
            }
        };

        let decoded_byte_length = view
            .get("byteLength")
            .and_then(Value::as_u64)
            .ok_or_else(|| missing("byteLength"))?;
        plan.views.push(MeshoptView {
            byte_offset: view
                .get("byteOffset")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            decoded_byte_length,
            source_buffer,
            ext_byte_offset: ext.get("byteOffset").and_then(Value::as_u64).unwrap_or(0),
            ext_byte_length: ext
                .get("byteLength")
                .and_then(Value::as_u64)
                .ok_or_else(|| missing("byteLength"))?,
            mode,
            count: ext.get("count").and_then(Value::as_u64).ok_or_else(|| missing("count"))?,
            stride: ext.get("byteStride").and_then(Value::as_u64).ok_or_else(|| missing("byteStride"))?,
            filter,
        });
        plan.decoded_len += decoded_byte_length;
    }
    for (i, buffer) in root
        .get("buffers")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let is_virtual = buffer
            .get("extensions")
            .and_then(|e| e.get(MESHOPT_EXT))
            .and_then(|e| e.get("fallback"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if is_virtual && !virtual_buffers.contains(&i) {
            virtual_buffers.push(i);
        }
    }
    Ok((plans, virtual_buffers))
}

/// Decodes an attributes view (`mode: ATTRIBUTES`) of the given vertex
/// stride. Only the strides gltfpack emits are supported (2, 4, 6, 8, 12,
/// 16, 32 bytes).
fn decode_attributes(src: &[u8], count: usize, stride: usize) -> Result<Vec<u8>, GltfError> {
    macro_rules! decode_with_stride {
        ($n:literal) => {{
            let blocks = decode_vertex_buffer::<[u8; $n]>(src, count)
                .map_err(|err| {
                    invalid(&format!("meshopt_decodeVertexBuffer failed: {err:?}"))
                })?;
            return Ok(blocks.into_iter().flatten().collect());
        }};
    }
    match stride {
        2 => decode_with_stride!(2),
        4 => decode_with_stride!(4),
        6 => decode_with_stride!(6),
        8 => decode_with_stride!(8),
        12 => decode_with_stride!(12),
        16 => decode_with_stride!(16),
        32 => decode_with_stride!(32),
        other => Err(invalid(&format!(
            "Unsupported meshopt vertex stride: {other}"
        ))),
    }
}

/// Decodes meshopt-compressed views into their virtual decoded buffers.
/// Call after the raw buffers are loaded and before any accessor data is
/// read.
pub(super) fn decode_meshopt_buffers(
    buffer_data: &mut Vec<Vec<u8>>,
    plans: &[MeshoptBufferPlan],
) -> Result<(), GltfError> {
    for plan in plans {
        let source = &buffer_data[plan.source_buffer];
        let mut decoded = vec![0u8; plan.decoded_len as usize];
        for view in &plan.views {
            let start = view.ext_byte_offset as usize;
            let end = start + view.ext_byte_length as usize;
            let src = source
                .get(start..end)
                .ok_or_else(|| invalid("Meshopt view range out of buffer bounds"))?;

            let out_len = (view.count * view.stride) as usize;
            let mut out = vec![0u8; out_len];

            // Only the attributes and triangles modes are supported; the
            // index-sequence mode and meshopt filters are not available
            // through the safe `meshopt` crate API.
            match view.mode {
                MeshoptMode::Attributes => {
                    out.copy_from_slice(&decode_attributes(src, view.count as usize, view.stride as usize)?);
                }
                MeshoptMode::Triangles => {
                    match view.stride {
                        2 => {
                            let indices: Vec<u16> =
                                decode_index_buffer(src, view.count as usize).map_err(|err| {
                                    invalid(&format!("meshopt_decodeIndexBuffer failed: {err:?}"))
                                })?;
                            for (dst, src) in out.chunks_exact_mut(2).zip(indices) {
                                dst.copy_from_slice(&src.to_le_bytes());
                            }
                        }
                        4 => {
                            let indices: Vec<u32> =
                                decode_index_buffer(src, view.count as usize).map_err(|err| {
                                    invalid(&format!("meshopt_decodeIndexBuffer failed: {err:?}"))
                                })?;
                            for (dst, src) in out.chunks_exact_mut(4).zip(indices) {
                                dst.copy_from_slice(&src.to_le_bytes());
                            }
                        }
                        other => {
                            return Err(invalid(&format!(
                                "Unsupported meshopt index stride: {other}"
                            )));
                        }
                    }
                }
            }

            let dst_start = view.byte_offset as usize;
            let dst_end = dst_start + out.len();
            let dst = decoded
                .get_mut(dst_start..dst_end)
                .ok_or_else(|| invalid("Meshopt view does not fit its decoded buffer"))?;
            dst.copy_from_slice(&out);
        }
        buffer_data[plan.buffer_index] = decoded;
    }
    Ok(())
}
