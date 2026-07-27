use crate::gguf::{GGUFReader, TensorType};
use crate::model::Model;
use crate::vulkan::{GpuBuffer, VulkanContext};
use ash::vk;

pub struct DFlashLayer {
    pub attn_norm_w: Option<GpuBuffer>,
    pub attn_q_w: Option<GpuBuffer>,
    pub attn_k_w: Option<GpuBuffer>,
    pub attn_v_w: Option<GpuBuffer>,
    pub attn_output_w: Option<GpuBuffer>,
    pub attn_q_norm_w: Option<GpuBuffer>,
    pub attn_k_norm_w: Option<GpuBuffer>,
    pub attn_gate_w: Option<GpuBuffer>,
    pub ffn_norm_w: Option<GpuBuffer>,
    pub ffn_gate_w: Option<GpuBuffer>,
    pub ffn_up_w: Option<GpuBuffer>,
    pub ffn_down_w: Option<GpuBuffer>,
}

#[allow(dead_code)]
pub struct DFlashModel {
    pub model: Model,
    pub layers: Vec<DFlashLayer>,
    pub tok_embeddings: Option<GpuBuffer>,
    pub output_norm_w: Option<GpuBuffer>,
    pub fc_weight: Option<GpuBuffer>,
    pub ctx: VulkanContext,
}

impl DFlashModel {
    pub fn load(gguf_path: &std::path::Path) -> Result<Self, String> {
        let reader =
            GGUFReader::open(gguf_path).map_err(|e| format!("failed to open model: {}", e))?;
        let mut model = Model::from_reader(reader);

        let ctx = VulkanContext::init();

        let block_count = model.block_count;
        let mut layers = Vec::with_capacity(block_count);

        for layer in 0..block_count {
            let mut l = DFlashLayer {
                attn_norm_w: None,
                attn_q_w: None,
                attn_k_w: None,
                attn_v_w: None,
                attn_output_w: None,
                attn_q_norm_w: None,
                attn_k_norm_w: None,
                attn_gate_w: None,
                ffn_norm_w: None,
                ffn_gate_w: None,
                ffn_up_w: None,
                ffn_down_w: None,
            };

            macro_rules! load_buf {
                ($lfield:ident, $suffix:literal) => {
                    if let Some(t) = model.block_tensor(layer, $suffix).cloned() {
                        l.$lfield = Some(model.upload_tensor_f32(&ctx, &t));
                    }
                };
            }

            load_buf!(attn_norm_w, "attn_norm.weight");
            load_buf!(attn_q_w, "attn_q.weight");
            load_buf!(attn_k_w, "attn_k.weight");
            load_buf!(attn_v_w, "attn_v.weight");
            load_buf!(attn_output_w, "attn_output.weight");
            load_buf!(attn_q_norm_w, "attn_q_norm.weight");
            load_buf!(attn_k_norm_w, "attn_k_norm.weight");
            load_buf!(attn_gate_w, "attn_gate.weight");
            load_buf!(ffn_norm_w, "ffn_norm.weight");
            load_buf!(ffn_gate_w, "ffn_gate.weight");
            load_buf!(ffn_up_w, "ffn_up.weight");
            load_buf!(ffn_down_w, "ffn_down.weight");

            layers.push(l);
        }

        let tok_embeddings = model
            .tensor("token_embd.weight")
            .cloned()
            .map(|t| model.upload_tensor_f32(&ctx, &t));

        let output_norm_w = model
            .tensor("output_norm.weight")
            .cloned()
            .map(|t| model.upload_tensor_f32(&ctx, &t));

        let fc_weight = model
            .tensor("fc.weight")
            .cloned()
            .map(|t| model.upload_tensor_f32(&ctx, &t));

        Ok(DFlashModel {
            model,
            layers,
            tok_embeddings,
            output_norm_w,
            fc_weight,
            ctx,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn embedding_length(&self) -> usize {
        self.model.embedding_length
    }

    pub fn ffn_length(&self) -> usize {
        self.model.ffn_length
    }

    pub fn head_count(&self) -> usize {
        self.model.head_count
    }

    pub fn head_count_kv(&self) -> usize {
        self.model.head_count_kv
    }

    pub fn rope_dim(&self) -> usize {
        self.model.rope_dim
    }

    pub fn head_dim(&self) -> usize {
        self.model.head_dim
    }

    pub fn target_layers(&self) -> &[i64] {
        &self.model.target_layers
    }
}

#[allow(dead_code)]
pub struct ForwardBuffers {
    pub x: GpuBuffer,
    pub norm_x: GpuBuffer,
    pub q: GpuBuffer,
    pub k: GpuBuffer,
    pub v: GpuBuffer,
    pub q_normed: GpuBuffer,
    pub k_normed: GpuBuffer,
    pub attn_out: GpuBuffer,
    pub ffn_gate: GpuBuffer,
    pub ffn_up: GpuBuffer,
    pub ffn_h: GpuBuffer,
    pub ffn_out: GpuBuffer,
}

impl ForwardBuffers {
    #[allow(dead_code)]
    pub fn new(
        ctx: &VulkanContext,
        n_embd: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_ff: usize,
    ) -> Self {
        let flags = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mem_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        Self {
            x: GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags),
            norm_x: GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags),
            q: GpuBuffer::new(ctx, (n_heads * head_dim * 4) as u64, flags, mem_flags),
            k: GpuBuffer::new(ctx, (n_kv_heads * head_dim * 4) as u64, flags, mem_flags),
            v: GpuBuffer::new(ctx, (n_kv_heads * head_dim * 4) as u64, flags, mem_flags),
            q_normed: GpuBuffer::new(ctx, (n_heads * head_dim * 4) as u64, flags, mem_flags),
            k_normed: GpuBuffer::new(ctx, (n_kv_heads * head_dim * 4) as u64, flags, mem_flags),
            attn_out: GpuBuffer::new(ctx, (n_heads * head_dim * 4) as u64, flags, mem_flags),
            ffn_gate: GpuBuffer::new(ctx, (n_ff * 4) as u64, flags, mem_flags),
            ffn_up: GpuBuffer::new(ctx, (n_ff * 4) as u64, flags, mem_flags),
            ffn_h: GpuBuffer::new(ctx, (n_ff * 4) as u64, flags, mem_flags),
            ffn_out: GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags),
        }
    }
}

#[allow(dead_code)]
pub fn upload_f32(ctx: &VulkanContext, data: &[f32]) -> GpuBuffer {
    let flags = vk::BufferUsageFlags::STORAGE_BUFFER;
    let mem_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let buf = GpuBuffer::new(ctx, (data.len() * 4) as u64, flags, mem_flags);
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    buf.upload(&bytes);
    buf
}

#[allow(dead_code)]
pub fn read_f32_sync(buf: &GpuBuffer, n: usize) -> Vec<f32> {
    buf.read_f32(n)
}

#[allow(dead_code)]
pub fn upload_f32_to(buf: &GpuBuffer, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    buf.upload(&bytes);
}

#[allow(dead_code)]
fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn dequant_to_f32(data: &[u8], dtype: TensorType, n_elements: usize) -> Vec<f32> {
    match dtype {
        TensorType::F32 => {
            let mut result = Vec::with_capacity(n_elements);
            for i in 0..n_elements {
                let offset = i * 4;
                if offset + 4 <= data.len() {
                    let bits = u32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    result.push(f32::from_bits(bits));
                } else {
                    result.push(0.0);
                }
            }
            result
        }
        TensorType::BF16 => {
            let mut result = Vec::with_capacity(n_elements);
            for i in 0..n_elements {
                let offset = i * 2;
                if offset + 2 <= data.len() {
                    let h_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                    result.push(f32::from_bits((h_bits as u32) << 16));
                } else {
                    result.push(0.0);
                }
            }
            result
        }
        TensorType::Q8_0 => {
            let n_blocks = n_elements.div_ceil(32);
            let mut result = vec![0.0f32; n_elements];
            for block_idx in 0..n_blocks {
                let offset = block_idx * 34;
                if offset + 34 > data.len() {
                    break;
                }
                let h_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                let d = if h_bits == 0 {
                    0.0
                } else {
                    f32::from_bits((h_bits as u32) << 16)
                };
                let start = block_idx * 32;
                let end = (start + 32).min(n_elements);
                for i in 0..(end - start) {
                    let q = data[offset + 2 + i] as i8 as f32;
                    result[start + i] = q * d;
                }
            }
            result
        }
        TensorType::Q4_K => {
            let n_blocks = n_elements.div_ceil(256);
            let mut result = vec![0.0f32; n_elements];
            for block_idx in 0..n_blocks {
                let base = block_idx * 144;
                if base + 144 > data.len() {
                    break;
                }
                let d_bits = u16::from_le_bytes([data[base], data[base + 1]]);
                let d = f32::from_bits((d_bits as u32) << 16);
                let k4_bits = u16::from_le_bytes([data[base + 2], data[base + 3]]);
                let k4 = f32::from_bits((k4_bits as u32) << 16);
                let signs_bits = u16::from_le_bytes([data[base + 4], data[base + 5]]);
                let signs = f32::from_bits((signs_bits as u32) << 16);
                let start = block_idx * 256;
                let end = (start + 256).min(n_elements);
                let mut sign_idx = 0.0;
                for i in 0..(end - start) {
                    let byte_idx = base + 6 + (i / 2);
                    if byte_idx >= base + 144 {
                        break;
                    }
                    let byte = data[byte_idx];
                    let q = if i % 2 == 0 {
                        (byte >> 4) & 0x07
                    } else {
                        byte & 0x07
                    };
                    let s = if signs < 0.0 { -1.0 } else { 1.0 };
                    result[start + i] = q as f32 * d + s * k4;
                    if i % 2 != 0 {
                        if signs < 0.0 {
                            sign_idx += 1.0;
                        }
                    }
                }
            }
            result
        }
        TensorType::F16 => {
            let mut result = Vec::with_capacity(n_elements);
            for i in 0..n_elements {
                let offset = i * 2;
                if offset + 2 <= data.len() {
                    let h_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                    let sign = if (h_bits & 0x8000) != 0 { -1.0f32 } else { 1.0 };
                    let exp = ((h_bits >> 10) & 0x1F) as i32;
                    let mant = (h_bits & 0x3FF) as f32;
                    if exp == 0 {
                        if mant == 0.0 {
                            result.push(sign * 0.0);
                        } else {
                            result.push(sign * 2f32.powi(-14) * (mant / 1024.0));
                        }
                    } else if exp == 31 {
                        result.push(if mant == 0.0 { sign * f32::INFINITY } else { f32::NAN });
                    } else {
                        result.push(sign * 2f32.powi(exp - 15) * (1.0 + mant / 1024.0));
                    }
                } else {
                    result.push(0.0);
                }
            }
            result
        }
        _ => vec![0.0f32; n_elements],
    }
}

#[allow(dead_code)]
impl DFlashModel {
    pub fn forward_token(&self, bufs: &ForwardBuffers, token_id: usize) -> Vec<f32> {
        let n_embd = self.embedding_length();
        let n_heads = self.head_count();
        let n_kv_heads = self.head_count_kv();
        let head_dim_per_head = self.head_dim();
        let n_ff = self.ffn_length();
        let rope_dim = self.rope_dim();

        let ctx = &self.ctx;

        if let Some(tok_emb) = &self.tok_embeddings {
            let vocab_size = tok_emb.size as usize / (n_embd * 4);
            if token_id < vocab_size {
                let all_embs = tok_emb.read_f32(tok_emb.size as usize / 4);
                let token_emb: Vec<f32> = (0..n_embd)
                    .map(|i| all_embs[i * vocab_size + token_id])
                    .collect();
                bufs.x.upload(&f32_to_bytes(&token_emb));
            } else {
                eprintln!("token_id {} >= vocab_size {}", token_id, vocab_size);
                std::process::exit(1);
            }
        } else {
            let dummy_input = vec![0.0f32; n_embd];
            bufs.x.upload(&f32_to_bytes(&dummy_input));
        }

        let rmsnorm_shader = ctx.create_shader_module(crate::include_spv!(rmsnorm));
        let proj_shader = ctx.create_shader_module(crate::include_spv!(proj));
        let repeat_v_shader = ctx.create_shader_module(crate::include_spv!(repeat_v));
        let qk_norm_shader = ctx.create_shader_module(crate::include_spv!(qk_norm));
        let rope_shader = ctx.create_shader_module(crate::include_spv!(rope));
        let activation_shader = ctx.create_shader_module(crate::include_spv!(activation));
        let gate_mul_shader = ctx.create_shader_module(crate::include_spv!(gate_mul));
        let add_shader = ctx.create_shader_module(crate::include_spv!(add));

        let freq: Vec<f32> = (0..rope_dim / 2)
            .map(|i| {
                let base = self
                    .model
                    .reader
                    .metadata
                    .rope_freq_base
                    .unwrap_or(500000.0);
                let dim = rope_dim as f32;
                1.0 / base.powf((2.0 * i as f32) / dim)
            })
            .collect();
        let freq_buf = upload_f32(ctx, &freq);

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let n_heads_x_hd = n_heads * head_dim_per_head;

            let attn_norm_w = layer.attn_norm_w.as_ref().expect("attn_norm_w");
            let attn_q_w = layer.attn_q_w.as_ref().expect("attn_q_w");
            let attn_k_w = layer.attn_k_w.as_ref().expect("attn_k_w");
            let attn_v_w = layer.attn_v_w.as_ref().expect("attn_v_w");
            let attn_output_w = layer.attn_output_w.as_ref().expect("attn_output_w");
            let attn_q_norm_w = layer.attn_q_norm_w.as_ref().expect("attn_q_norm_w");
            let attn_k_norm_w = layer.attn_k_norm_w.as_ref().expect("attn_k_norm_w");
            let attn_gate_w = layer.attn_gate_w.as_ref().expect("attn_gate_w");
            let ffn_norm_w = layer.ffn_norm_w.as_ref().expect("ffn_norm_w");
            let ffn_gate_w = layer.ffn_gate_w.as_ref().expect("ffn_gate_w");
            let ffn_up_w = layer.ffn_up_w.as_ref().expect("ffn_up_w");
            let ffn_down_w = layer.ffn_down_w.as_ref().expect("ffn_down_w");

            dispatch_rmsnorm(
                ctx,
                &rmsnorm_shader,
                attn_norm_w,
                &bufs.x,
                &bufs.norm_x,
                n_embd,
            );

            dispatch_proj(
                ctx,
                &proj_shader,
                &bufs.norm_x,
                attn_q_w,
                &bufs.q,
                n_embd,
                n_heads_x_hd,
            );
            dispatch_proj(
                ctx,
                &proj_shader,
                &bufs.norm_x,
                attn_k_w,
                &bufs.k,
                n_embd,
                n_kv_heads * head_dim_per_head,
            );
            dispatch_proj(
                ctx,
                &proj_shader,
                &bufs.norm_x,
                attn_v_w,
                &bufs.v,
                n_embd,
                n_kv_heads * head_dim_per_head,
            );

            let push_ropec = [
                n_heads as u32,
                head_dim_per_head as u32,
                1u32,
                rope_dim as u32,
            ];
            dispatch_qk_norm(
                ctx,
                &qk_norm_shader,
                &bufs.q,
                attn_q_norm_w,
                &bufs.q_normed,
                n_heads,
                head_dim_per_head,
                n_heads_x_hd,
            );
            dispatch_qk_norm(
                ctx,
                &qk_norm_shader,
                &bufs.k,
                attn_k_norm_w,
                &bufs.k_normed,
                n_kv_heads,
                head_dim_per_head,
                n_kv_heads * head_dim_per_head,
            );

            dispatch_rope(
                ctx,
                &rope_shader,
                &bufs.q_normed,
                &freq_buf,
                &bufs.q,
                &push_ropec,
            );
            dispatch_rope(
                ctx,
                &rope_shader,
                &bufs.k_normed,
                &freq_buf,
                &bufs.k,
                &push_ropec,
            );

            let v_rep_out = GpuBuffer::new(
                ctx,
                (n_heads_x_hd * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            dispatch_repeat_v(
                ctx,
                &repeat_v_shader,
                &bufs.v,
                &v_rep_out,
                n_kv_heads * head_dim_per_head,
                n_heads_x_hd,
            );

            let gate_data =
                dispatch_proj_read(ctx, &proj_shader, &bufs.norm_x, attn_gate_w, n_embd, n_heads);

            let gate_f32: Vec<f32> = gate_data
                .iter()
                .map(|&g| if g > 20.0 { g } else { (1.0 + g.exp()).ln() })
                .collect();
            let gate_bytes = f32_to_bytes(&gate_f32);
            let gate_buf = GpuBuffer::new(
                ctx,
                (n_heads * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            gate_buf.upload(&gate_bytes);

            dispatch_gate_mul(
                ctx,
                &gate_mul_shader,
                &bufs.attn_out,
                &v_rep_out,
                &gate_buf,
                n_heads_x_hd,
                n_heads,
                head_dim_per_head,
            );

            dispatch_proj(
                ctx,
                &proj_shader,
                &bufs.attn_out,
                attn_output_w,
                &bufs.ffn_out,
                n_heads_x_hd,
                n_embd,
            );

            dispatch_add(ctx, &add_shader, &bufs.x, &bufs.ffn_out, &bufs.x, n_embd);

            dispatch_rmsnorm(
                ctx,
                &rmsnorm_shader,
                ffn_norm_w,
                &bufs.x,
                &bufs.norm_x,
                n_embd,
            );

            dispatch_proj(
                ctx,
                &proj_shader,
                &bufs.norm_x,
                ffn_gate_w,
                &bufs.ffn_gate,
                n_embd,
                n_ff,
            );
            dispatch_proj(
                ctx,
                &proj_shader,
                &bufs.norm_x,
                ffn_up_w,
                &bufs.ffn_up,
                n_embd,
                n_ff,
            );

            dispatch_activation(
                ctx,
                &activation_shader,
                &bufs.ffn_gate,
                &bufs.ffn_up,
                &bufs.ffn_h,
                n_ff,
            );

            dispatch_proj(
                ctx,
                &proj_shader,
                &bufs.ffn_h,
                ffn_down_w,
                &bufs.ffn_out,
                n_ff,
                n_embd,
            );
            dispatch_add(ctx, &add_shader, &bufs.x, &bufs.ffn_out, &bufs.x, n_embd);

        }

        if let Some(output_norm_w) = &self.output_norm_w {
            let push: [u8; 12] = {
                let mut b = [0u8; 12];
                b[..4].copy_from_slice(&(n_embd as u32).to_le_bytes());
                let eps: f32 = 1e-5;
                b[4..8].copy_from_slice(&eps.to_le_bytes());
                b[8..].copy_from_slice(&0u32.to_le_bytes());
                b
            };
            dispatch_rmsnorm(
                ctx,
                &rmsnorm_shader,
                output_norm_w,
                &bufs.x,
                &bufs.x,
                &push,
                n_embd,
            );
        }

        if let Some(fc_w) = &self.fc_weight {
            let vocab_size = fc_w.size as usize / (n_embd * 4);
            let logits_buf = GpuBuffer::new(
                ctx,
                (vocab_size * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            dispatch_proj(
                ctx,
                &proj_shader,
                &bufs.x,
                fc_w,
                &logits_buf,
                n_embd,
                vocab_size,
            );
            logits_buf.read_f32(vocab_size)
        } else {
            bufs.x.read_f32(n_embd)
        }
    }
}

fn dispatch_rmsnorm(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    w: &GpuBuffer,
    x: &GpuBuffer,
    out: &GpuBuffer,
    n: usize,
) {
    let eps: f32 = 1e-5;
    let push: [u8; 8] = {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&(n as u32).to_le_bytes());
        b[4..].copy_from_slice(&eps.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, w.buffer, w.size);
    ctx.write_buffer_descriptor(set, 2, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 8);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

fn dispatch_proj(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    x: &GpuBuffer,
    w: &GpuBuffer,
    c: &GpuBuffer,
    k: usize,
    n: usize,
) {
    let push: [u8; 8] = {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&(k as u32).to_le_bytes());
        b[4..].copy_from_slice(&(n as u32).to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, w.buffer, w.size);
    ctx.write_buffer_descriptor(set, 2, c.buffer, c.size);

    let layout = ctx.create_pipeline_layout(set_layout, 8);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_proj_read(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    x: &GpuBuffer,
    w: &GpuBuffer,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let result_buf = GpuBuffer::new(
        ctx,
        (n * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    let push: [u8; 8] = {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&(k as u32).to_le_bytes());
        b[4..].copy_from_slice(&(n as u32).to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, w.buffer, w.size);
    ctx.write_buffer_descriptor(set, 2, result_buf.buffer, result_buf.size);

    let layout = ctx.create_pipeline_layout(set_layout, 8);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));

    result_buf.read_f32(n)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_qk_norm(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    q: &GpuBuffer,
    norm_w: &GpuBuffer,
    out: &GpuBuffer,
    n_heads: usize,
    head_dim: usize,
    n_total: usize,
) {
    let eps: f32 = 1e-6;
    let push: [u8; 16] = {
        let mut b = [0u8; 16];
        b[..4].copy_from_slice(&(n_heads as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(head_dim as u32).to_le_bytes());
        b[8..12].copy_from_slice(&(n_total as u32).to_le_bytes());
        b[12..].copy_from_slice(&eps.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, q.buffer, q.size);
    ctx.write_buffer_descriptor(set, 1, norm_w.buffer, norm_w.size);
    ctx.write_buffer_descriptor(set, 2, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 16);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n_total.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

fn dispatch_rope(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    x: &GpuBuffer,
    freq: &GpuBuffer,
    out: &GpuBuffer,
    push: &[u32; 4],
) {
    let push_bytes: [u8; 16] = {
        let mut b = [0u8; 16];
        b[..4].copy_from_slice(&push[0].to_le_bytes());
        b[4..8].copy_from_slice(&push[1].to_le_bytes());
        b[8..12].copy_from_slice(&push[2].to_le_bytes());
        b[12..].copy_from_slice(&push[3].to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, freq.buffer, freq.size);
    ctx.write_buffer_descriptor(set, 2, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 16);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let total = push[0] as usize * push[1] as usize * push[2] as usize;
    let groups = total.div_ceil(256);
    ctx.submit_compute(pipeline, layout, set, &push_bytes, (groups, 1, 1));
}

fn dispatch_repeat_v(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    v: &GpuBuffer,
    out: &GpuBuffer,
    n_kv: usize,
    n_out: usize,
) {
    let push: [u8; 8] = {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&(n_kv as u32).to_le_bytes());
        b[4..].copy_from_slice(&(n_out as u32).to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(2);
    let pool = ctx.create_descriptor_pool(2);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, v.buffer, v.size);
    ctx.write_buffer_descriptor(set, 1, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 8);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n_out.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gate_mul(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    attn_out: &GpuBuffer,
    v_rep: &GpuBuffer,
    gate: &GpuBuffer,
    n_hd: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let push: [u8; 16] = {
        let mut b = [0u8; 16];
        b[..4].copy_from_slice(&(n_hd as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(n_heads as u32).to_le_bytes());
        b[8..12].copy_from_slice(&(head_dim as u32).to_le_bytes());
        b[12..].copy_from_slice(&0u32.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, v_rep.buffer, v_rep.size);
    ctx.write_buffer_descriptor(set, 1, gate.buffer, gate.size);
    ctx.write_buffer_descriptor(set, 2, attn_out.buffer, attn_out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 16);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n_hd.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

fn dispatch_proj_matmul(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    x: &GpuBuffer,
    w: &GpuBuffer,
    c: &GpuBuffer,
    k: usize,
    n: usize,
) {
    dispatch_proj(ctx, shader, x, w, c, k, n);
}

fn dispatch_add(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    a: &GpuBuffer,
    b: &GpuBuffer,
    out: &GpuBuffer,
    n: usize,
) {
    let push: [u8; 4] = (n as u32).to_le_bytes();

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, a.buffer, a.size);
    ctx.write_buffer_descriptor(set, 1, b.buffer, b.size);
    ctx.write_buffer_descriptor(set, 2, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 4);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

fn dispatch_activation(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    h: &GpuBuffer,
    n: usize,
) {
    let push: [u8; 4] = (n as u32).to_le_bytes();

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, gate.buffer, gate.size);
    ctx.write_buffer_descriptor(set, 1, up.buffer, up.size);
    ctx.write_buffer_descriptor(set, 2, h.buffer, h.size);

    let layout = ctx.create_pipeline_layout(set_layout, 4);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_bf16() {
        let data: Vec<u8> = (0..64)
            .flat_map(|i| {
                let v = (i as f32 - 32.0) * 0.001;
                let bits = (v.to_bits() >> 16) as u16;
                bits.to_le_bytes().to_vec()
            })
            .collect();
        let result = dequant_to_f32(&data, TensorType::BF16, 64);
        for i in 0..64 {
            let expected = (i as f32 - 32.0) * 0.001;
            assert!((result[i] - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn test_dequant_q8_0() {
        let mut data = vec![0u8; 34];
        let h_bits = (0.5f32.bits() >> 16) as u16;
        data[0..2].copy_from_slice(&h_bits.to_le_bytes());
        for i in 0..32 {
            data[2 + i] = 1;
        }
        let result = dequant_to_f32(&data, TensorType::Q8_0, 32);
        assert!((result[0] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_dequant_f32() {
        let data: Vec<u8> = (0..16)
            .flat_map(|i| (i as f32 * 0.001).to_le_bytes().to_vec())
            .collect();
        let result = dequant_to_f32(&data, TensorType::F32, 16);
        assert!((result[5] - 0.005).abs() < 1e-6);
    }

    #[test]
    fn test_dequant_unsupported() {
        let data = vec![0u8; 32];
        let result = dequant_to_f32(&data, TensorType::Q5_0, 16);
        assert_eq!(result.len(), 16);
        assert!(result.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_bf16_roundtrip() {
        let original: f32 = 3.14159;
        let bits = (original.to_bits() >> 16) as u16;
        let restored = f32::from_bits((bits as u32) << 16);
        assert!((original - restored).abs() < 1e-3);
    }
}
