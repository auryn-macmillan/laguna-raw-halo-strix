use std::path::Path;
use crate::gguf::{GGUFReader, TensorInfo};
use ash::vk;

#[allow(dead_code)]
pub struct Model {
    pub reader: GGUFReader,
    pub block_count: usize,
    pub embedding_length: usize,
    pub ffn_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub rope_dim: usize,
    pub rope_dim_swa: usize,
    pub head_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub expert_feed_forward_length: usize,
    pub shared_ffn_length: usize,
    pub has_moe: bool,
    pub has_shexp: bool,
    pub target_layers: Vec<i64>,
    pub is_dflash: bool,
}

#[allow(dead_code)]
impl Model {
    pub fn from_reader(reader: GGUFReader) -> Self {
        let m = &reader.metadata;
        let block_count = m.block_count.unwrap_or(0) as usize;
        let embedding_length = m.embedding_length.unwrap_or(0) as usize;
        let ffn_length = m.feed_forward_length.unwrap_or(0) as usize;
        let head_count = m.attention_head_count.unwrap_or(0) as usize;
        let head_count_kv = m.attention_head_count_kv.unwrap_or(0) as usize;
        let rope_dim = m.rope_dimension_count.unwrap_or(0) as usize;
        let rope_dim_swa = m.rope_dimension_count_swa.unwrap_or(0) as usize;
        let head_dim = m.attention_key_length.unwrap_or(0) as usize;
        let expert_count = m.expert_count.unwrap_or(0) as usize;
        let expert_used = m.expert_used_count.unwrap_or(0) as usize;
        let expert_ffn_meta = m.expert_feed_forward_length.unwrap_or(0) as usize;
        let shared_ffn_meta = m.expert_shared_feed_forward_length.unwrap_or(0) as usize;

        let expert_ffn_from_shape = reader
            .tensors
            .iter()
            .filter_map(|t| {
                if (t.name.contains("ffn_gate_exps") || t.name.contains("ffn_gate_shexp"))
                    && t.shape.len() >= 2
                {
                    Some(t.shape[1] as usize)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);
        let expert_ffn = if expert_ffn_meta > 0 {
            expert_ffn_meta
        } else {
            expert_ffn_from_shape
        };
        let shared_ffn = if shared_ffn_meta > 0 {
            shared_ffn_meta
        } else {
            expert_ffn_from_shape
        };

        let has_moe = reader
            .tensors
            .iter()
            .any(|t| t.name.contains("ffn_gate_exps"));
        let has_shexp = reader
            .tensors
            .iter()
            .any(|t| t.name.contains("ffn_gate_shexp"));

        let target_layers = m.target_layers.clone().unwrap_or_default();
        let is_dflash = m.architecture.as_deref() == Some("dflash");

        let expert_from_shape = reader
            .tensors
            .iter()
            .filter_map(|t| {
                if t.name.contains("ffn_gate_exps")
                    && t.shape.len() == 3
                    && t.shape.last().is_some()
                {
                    Some(*t.shape.last().unwrap() as usize)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);
        let n_expert = if expert_count > 0 {
            expert_count
        } else {
            expert_from_shape
        };
        let n_expert_used = if expert_used > 0 {
            expert_used
        } else if expert_from_shape > 0 {
            10
        } else {
            0
        };

        Self {
            reader,
            block_count,
            embedding_length,
            ffn_length,
            head_count,
            head_count_kv,
            rope_dim,
            rope_dim_swa,
            head_dim,
            n_expert,
            n_expert_used,
            n_ff_exp: expert_ffn,
            expert_feed_forward_length: expert_ffn,
            shared_ffn_length: shared_ffn,
            has_moe,
            has_shexp,
            target_layers,
            is_dflash,
        }
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.reader.tensor_by_name(name)
    }

    pub fn block_tensor(&self, block: usize, suffix: &str) -> Option<&TensorInfo> {
        let name = format!("blk.{}.{}", block, suffix);
        self.reader.tensor_by_name(&name)
    }

    pub fn load_tensor(&mut self, tensor: &TensorInfo) -> Vec<u8> {
        self.reader.read_tensor_data(tensor).unwrap_or_default()
    }

    pub fn upload_tensor(&mut self, ctx: &crate::vulkan::VulkanContext, tensor: &TensorInfo) -> crate::vulkan::GpuBuffer {
        let data = self.load_tensor(tensor);
        let byte_len = data.len() as u64;
        let buffer = crate::vulkan::GpuBuffer::new(
            ctx,
            byte_len,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        buffer.upload(&data);
        buffer
    }

    pub fn upload_tensor_f32(&mut self, ctx: &crate::vulkan::VulkanContext, tensor: &TensorInfo) -> crate::vulkan::GpuBuffer {
        use crate::dflash::dequant_to_f32;
        let data = self.load_tensor(tensor);
        let n_elements = tensor.n_elements() as usize;
        let floats = dequant_to_f32(&data, tensor.dtype, n_elements);
        let byte_len = floats.len() * 4;
        let buffer = crate::vulkan::GpuBuffer::new(
            ctx,
            byte_len as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let bytes: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
        buffer.upload(&bytes);
        buffer
    }

    pub fn upload_tensor_quantized(&mut self, ctx: &crate::vulkan::VulkanContext, tensor: &TensorInfo) -> crate::vulkan::GpuBuffer {
        let data = self.load_tensor(tensor);
        let buffer = crate::vulkan::GpuBuffer::new(
            ctx,
            data.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        buffer.upload(&data);
        buffer
    }
}
