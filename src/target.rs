use crate::gguf::{GGUFReader, TensorType};
use crate::model::Model;
use crate::vulkan::{GpuBuffer, VulkanContext};
use crate::dflash::upload_f32;
use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightType {
    F32,
    Q8_0,
    Q4_K,
}

impl WeightType {
    fn from_tensor_type(t: TensorType) -> Self {
        match t {
            TensorType::Q8_0 => WeightType::Q8_0,
            TensorType::Q4_K => WeightType::Q4_K,
            _ => WeightType::F32,
        }
    }
}

pub struct TargetWeight {
    pub buf: GpuBuffer,
    pub dtype: WeightType,
    pub n_elements: usize,
}

pub struct TargetDenseLayer {
    pub attn_norm_w: Option<TargetWeight>,
    pub attn_q_norm_w: Option<TargetWeight>,
    pub attn_k_norm_w: Option<TargetWeight>,
    pub attn_q_w: Option<TargetWeight>,
    pub attn_k_w: Option<TargetWeight>,
    pub attn_v_w: Option<TargetWeight>,
    pub attn_output_w: Option<TargetWeight>,
    pub attn_gate_w: Option<TargetWeight>,
    pub ffn_norm_w: Option<TargetWeight>,
    pub ffn_gate_w: Option<TargetWeight>,
    pub ffn_up_w: Option<TargetWeight>,
    pub ffn_down_w: Option<TargetWeight>,
}

pub struct TargetMoELayer {
    pub attn_norm_w: Option<TargetWeight>,
    pub attn_q_norm_w: Option<TargetWeight>,
    pub attn_k_norm_w: Option<TargetWeight>,
    pub attn_q_w: Option<TargetWeight>,
    pub attn_k_w: Option<TargetWeight>,
    pub attn_v_w: Option<TargetWeight>,
    pub attn_output_w: Option<TargetWeight>,
    pub attn_gate_w: Option<TargetWeight>,
    pub ffn_norm_w: Option<TargetWeight>,
    pub exp_probs_b_bias: Option<TargetWeight>,
    pub ffn_gate_inp_w: Option<TargetWeight>,
    pub ffn_gate_exps_w: Option<TargetWeight>,
    pub ffn_up_exps_w: Option<TargetWeight>,
    pub ffn_down_exps_w: Option<TargetWeight>,
    pub ffn_gate_shexp_w: Option<TargetWeight>,
    pub ffn_up_shexp_w: Option<TargetWeight>,
    pub ffn_down_shexp_w: Option<TargetWeight>,
}

pub struct TargetLayer {
    pub dense: TargetDenseLayer,
    pub moe: Option<TargetMoELayer>,
    pub is_moe: bool,
    pub n_q_heads: usize,
}

#[allow(dead_code)]
pub struct TargetModel {
    pub model: Model,
    pub layers: Vec<TargetLayer>,
    pub tok_embeddings: Option<TargetWeight>,
    pub output_norm_w: Option<TargetWeight>,
    pub output_weight: Option<TargetWeight>,
    pub ctx: VulkanContext,
}

impl TargetModel {
    fn load_weight(model: &mut Model, ctx: &VulkanContext, t: &crate::gguf::TensorInfo) -> TargetWeight {
        let n_elements = t.n_elements() as usize;
        let dtype = WeightType::from_tensor_type(t.dtype);
        match dtype {
            WeightType::F32 => {
                let buf = model.upload_tensor_f32(ctx, t);
                TargetWeight { buf, dtype, n_elements }
            }
            WeightType::Q8_0 | WeightType::Q4_K => {
                let buf = model.upload_tensor_quantized(ctx, t);
                TargetWeight { buf, dtype, n_elements }
            }
        }
    }

    pub fn load(gguf_path: &std::path::Path) -> Result<Self, String> {
        let reader =
            GGUFReader::open(gguf_path).map_err(|e| format!("failed to open model: {}", e))?;
        let mut model = Model::from_reader(reader);

        let ctx = VulkanContext::init();

        let block_count = model.block_count;
        let mut layers = Vec::with_capacity(block_count);

        for layer_idx in 0..block_count {
            let is_moe = model.has_moe && layer_idx > 0;

            let mut dense = TargetDenseLayer {
                attn_norm_w: None,
                attn_q_norm_w: None,
                attn_k_norm_w: None,
                attn_q_w: None,
                attn_k_w: None,
                attn_v_w: None,
                attn_output_w: None,
                attn_gate_w: None,
                ffn_norm_w: None,
                ffn_gate_w: None,
                ffn_up_w: None,
                ffn_down_w: None,
            };

            macro_rules! load_dense_w {
                ($lfield:ident, $suffix:literal) => {
                    if let Some(t) = model.block_tensor(layer_idx, $suffix).cloned() {
                        dense.$lfield = Some(Self::load_weight(&mut model, &ctx, &t));
                    }
                };
            }

            load_dense_w!(attn_norm_w, "attn_norm.weight");
            load_dense_w!(attn_q_norm_w, "attn_q_norm.weight");
            load_dense_w!(attn_k_norm_w, "attn_k_norm.weight");
            load_dense_w!(attn_q_w, "attn_q.weight");
            load_dense_w!(attn_k_w, "attn_k.weight");
            load_dense_w!(attn_v_w, "attn_v.weight");
            load_dense_w!(attn_output_w, "attn_output.weight");
            load_dense_w!(attn_gate_w, "attn_gate.weight");
            load_dense_w!(ffn_norm_w, "ffn_norm.weight");

            if !is_moe {
                load_dense_w!(ffn_gate_w, "ffn_gate.weight");
                load_dense_w!(ffn_up_w, "ffn_up.weight");
                load_dense_w!(ffn_down_w, "ffn_down.weight");
            }

            let moe = if is_moe {
                let mut m = TargetMoELayer {
                    attn_norm_w: None,
                    attn_q_norm_w: None,
                    attn_k_norm_w: None,
                    attn_q_w: None,
                    attn_k_w: None,
                    attn_v_w: None,
                    attn_output_w: None,
                    attn_gate_w: None,
                    ffn_norm_w: None,
                    exp_probs_b_bias: None,
                    ffn_gate_exps_w: None,
                    ffn_up_exps_w: None,
                    ffn_down_exps_w: None,
                    ffn_gate_inp_w: None,
                    ffn_gate_shexp_w: None,
                    ffn_up_shexp_w: None,
                    ffn_down_shexp_w: None,
                };

                macro_rules! load_moe_w {
                    ($lfield:ident, $suffix:literal) => {
                        if let Some(t) = model.block_tensor(layer_idx, $suffix).cloned() {
                            m.$lfield = Some(Self::load_weight(&mut model, &ctx, &t));
                        }
                    };
                }

                load_moe_w!(exp_probs_b_bias, "exp_probs_b.bias");
                load_moe_w!(ffn_gate_inp_w, "ffn_gate_inp.weight");
                load_moe_w!(ffn_gate_exps_w, "ffn_gate_exps.weight");
                load_moe_w!(ffn_up_exps_w, "ffn_up_exps.weight");
                load_moe_w!(ffn_down_exps_w, "ffn_down_exps.weight");
                load_moe_w!(ffn_gate_shexp_w, "ffn_gate_shexp.weight");
                load_moe_w!(ffn_up_shexp_w, "ffn_up_shexp.weight");
                load_moe_w!(ffn_down_shexp_w, "ffn_down_shexp.weight");

                Some(m)
            } else {
                None
            };

            let n_q_heads = model
                .block_tensor(layer_idx, "attn_q.weight")
                .and_then(|t| {
                    if t.shape.len() >= 2 {
                        Some(t.shape[1] as usize / model.head_dim)
                    } else {
                        Some(model.head_count)
                    }
                })
                .unwrap_or(model.head_count);

            layers.push(TargetLayer { dense, moe, is_moe, n_q_heads });
        }

        let tok_embeddings = model
            .tensor("token_embd.weight")
            .cloned()
            .map(|t| Self::load_weight(&mut model, &ctx, &t));

        let output_norm_w = model
            .tensor("output_norm.weight")
            .cloned()
            .map(|t| Self::load_weight(&mut model, &ctx, &t));

        let output_weight = model
            .tensor("output.weight")
            .cloned()
            .map(|t| Self::load_weight(&mut model, &ctx, &t));

        Ok(TargetModel {
            model,
            layers,
            tok_embeddings,
            output_norm_w,
            output_weight,
            ctx,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn embedding_length(&self) -> usize {
        self.model.embedding_length
    }

    pub fn n_expert(&self) -> usize {
        self.model.n_expert
    }

    pub fn n_expert_used(&self) -> usize {
        self.model.n_expert_used
    }

    pub fn n_ff_exp(&self) -> usize {
        self.model.n_ff_exp
    }

    pub fn target_layers(&self) -> &[i64] {
        &self.model.target_layers
    }

    pub fn vocab_size(&self) -> usize {
        let emb = &self.tok_embeddings.as_ref().expect("tok_embeddings");
        let n_embd = self.embedding_length();
        (emb.n_elements) / n_embd
    }

    pub fn read_tensor_raw(&mut self, name: &str) -> Option<Vec<u8>> {
        let tensor = self.model.tensor(name)?.clone();
        Some(self.model.load_tensor(&tensor))
    }

    pub fn forward_token(&self, input: &[f32], layer_indices: Option<&[usize]>) -> (Vec<f32>, Vec<Vec<f32>>) {
        let ctx = &self.ctx;
        let n_embd = self.embedding_length();
        let n_heads = self.model.head_count;
        let n_kv_heads = self.model.head_count_kv;
        let head_dim = self.model.head_dim;
        let rope_dim = self.model.rope_dim;
        let rope_dim_swa = self.model.rope_dim_swa;
        let swa = rope_dim_swa > 0 || self.model.reader.metadata.attention_sliding_window.unwrap_or(0) > 0;

        let rmsnorm_shader = ctx.create_shader_module(crate::include_spv!(rmsnorm));
        let proj_shader = ctx.create_shader_module(crate::include_spv!(proj));
        let proj_q8_0_shader = ctx.create_shader_module(crate::include_spv!(proj_q8_0));
        let proj_q4k_shader = ctx.create_shader_module(crate::include_spv!(proj_q4k));
        let mha_attn_shader = ctx.create_shader_module(crate::include_spv!(mha_attn));
        let gate_attn_shader = ctx.create_shader_module(crate::include_spv!(gate_attn));
        let repeat_v_shader = ctx.create_shader_module(crate::include_spv!(repeat_v));
        let qk_norm_rope_shader = ctx.create_shader_module(crate::include_spv!(qk_norm_rope));
        let qk_norm_shader = ctx.create_shader_module(crate::include_spv!(qk_norm));
        let qk_norm_rope_swa_shader = ctx.create_shader_module(crate::include_spv!(qk_norm_rope));
        let activation_shader = ctx.create_shader_module(crate::include_spv!(activation));
        let add_shader = ctx.create_shader_module(crate::include_spv!(add));

        let _ = &proj_q8_0_shader; // suppress unused warning; will be used in dispatch_proj_quant

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

        let freq_swa: Vec<f32> = if swa {
            (0..rope_dim_swa / 2)
                .map(|i| {
                    let base = self
                        .model
                        .reader
                        .metadata
                        .rope_freq_base_swa
                        .unwrap_or(10000.0);
                    let dim = rope_dim_swa as f32;
                    1.0 / base.powf((2.0 * i as f32) / dim)
                })
                .collect()
        } else {
            vec![]
        };
        let freq_swa_buf = if swa { Some(upload_f32(ctx, &freq_swa)) } else { None };

        let flags = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mem_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let max_n_heads = self.layers.iter().map(|l| l.n_q_heads).max().unwrap_or(n_heads);
        let max_qkv = max_n_heads * head_dim;

        let x_buf = GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags);
        let norm_x_buf = GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags);
        let q_buf = GpuBuffer::new(ctx, (max_qkv * 4) as u64, flags, mem_flags);
        let k_buf = GpuBuffer::new(ctx, (n_kv_heads * head_dim * 4) as u64, flags, mem_flags);
        let v_buf = GpuBuffer::new(ctx, (n_kv_heads * head_dim * 4) as u64, flags, mem_flags);
        let q_normed_buf = GpuBuffer::new(ctx, (max_qkv * 4) as u64, flags, mem_flags);
        let k_normed_buf = GpuBuffer::new(ctx, (n_kv_heads * head_dim * 4) as u64, flags, mem_flags);
        let attn_out_buf = GpuBuffer::new(ctx, (max_qkv * 4) as u64, flags, mem_flags);
        let ffn_gate_buf = GpuBuffer::new(ctx, (self.model.ffn_length * 4) as u64, flags, mem_flags);
        let ffn_up_buf = GpuBuffer::new(ctx, (self.model.ffn_length * 4) as u64, flags, mem_flags);
        let ffn_h_buf = GpuBuffer::new(ctx, (self.model.ffn_length * 4) as u64, flags, mem_flags);
        let ffn_out_buf = GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags);


        let mut captured_states: Vec<Vec<f32>> = Vec::new();
        let capture_set = layer_indices.unwrap_or(&[]);
        let capture_all = layer_indices.is_none();

        let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        x_buf.upload(&input_bytes);

        for layer_idx in 0..self.layers.len() {
            if capture_all || capture_set.contains(&layer_idx) {
                captured_states.push(x_buf.read_f32(n_embd));
            }

            let layer = &self.layers[layer_idx];
            let n_q_heads = layer.n_q_heads;
            let n_heads_x_hd = n_q_heads * head_dim;

            let attn_norm_w = layer.dense.attn_norm_w.as_ref().expect("attn_norm_w");
            let attn_q_norm_w = layer.dense.attn_q_norm_w.as_ref().expect("attn_q_norm_w");
            let attn_k_norm_w = layer.dense.attn_k_norm_w.as_ref().expect("attn_k_norm_w");
            let attn_q_w = layer.dense.attn_q_w.as_ref().expect("attn_q_w");
            let attn_k_w = layer.dense.attn_k_w.as_ref().expect("attn_k_w");
            let attn_v_w = layer.dense.attn_v_w.as_ref().expect("attn_v_w");
            let attn_output_w = layer.dense.attn_output_w.as_ref().expect("attn_output_w");
            let ffn_norm_w = layer.dense.ffn_norm_w.as_ref().expect("ffn_norm_w");

            dispatch_rmsnorm(
                ctx, &rmsnorm_shader, &attn_norm_w.buf, &x_buf, &norm_x_buf, n_embd,
            );

            dispatch_proj_quant(
                ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader, &norm_x_buf, &attn_q_w.buf, &q_buf,
                n_embd, n_heads_x_hd, attn_q_w.dtype,
            );
            dispatch_proj_quant(
                ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader, &norm_x_buf, &attn_k_w.buf, &k_buf,
                n_embd, n_kv_heads * head_dim, attn_k_w.dtype,
            );
            dispatch_proj_quant(
                ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader, &norm_x_buf, &attn_v_w.buf, &v_buf,
                n_embd, n_kv_heads * head_dim, attn_v_w.dtype,
            );

            let layer_is_swa = swa && (layer_idx % 4 != 0);

            if layer_is_swa {
                dispatch_qk_norm_rope_swa(
                    ctx, &qk_norm_rope_swa_shader, &q_buf, &attn_q_norm_w.buf,
                    freq_swa_buf.as_ref().unwrap(), &q_normed_buf,
                    n_q_heads, head_dim, rope_dim_swa, 0,
                );
                dispatch_qk_norm_rope_swa(
                    ctx, &qk_norm_rope_swa_shader, &k_buf, &attn_k_norm_w.buf,
                    freq_swa_buf.as_ref().unwrap(), &k_normed_buf,
                    n_kv_heads, head_dim, rope_dim_swa, 0,
                );
            } else {
                dispatch_qk_norm_rope(
                    ctx, &qk_norm_rope_shader, &q_buf, &attn_q_norm_w.buf,
                    &freq_buf, &q_normed_buf, n_q_heads, head_dim, rope_dim, 0,
                );
                dispatch_qk_norm_rope(
                    ctx, &qk_norm_rope_shader, &k_buf, &attn_k_norm_w.buf,
                    &freq_buf, &k_normed_buf, n_kv_heads, head_dim, rope_dim, 0,
                );
            }

            let v_rep_out = GpuBuffer::new(
                ctx, (n_heads_x_hd * 4) as u64, flags, mem_flags,
            );

            dispatch_mha_attn(
                ctx, &mha_attn_shader, &q_normed_buf, &k_normed_buf, &v_buf,
                &attn_out_buf, n_q_heads, n_kv_heads, head_dim,
            );

            if let Some(attn_gate_w) = &layer.dense.attn_gate_w {
                let gate_buf = GpuBuffer::new(ctx, (n_q_heads * 4) as u64, flags, mem_flags);
                dispatch_proj_quant(
                    ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader,
                    &norm_x_buf, &attn_gate_w.buf, &gate_buf,
                    n_embd, n_q_heads, attn_gate_w.dtype,
                );
                dispatch_gate_attn(
                    ctx, &gate_attn_shader, &attn_out_buf, &gate_buf,
                    &attn_out_buf, n_q_heads, head_dim,
                );
            }

            dispatch_proj_quant(
                ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader, &attn_out_buf, &attn_output_w.buf, &ffn_out_buf,
                n_heads_x_hd, n_embd, attn_output_w.dtype,
            );
            dispatch_add(ctx, &add_shader, &x_buf, &ffn_out_buf, &x_buf, n_embd);

            dispatch_rmsnorm(
                ctx, &rmsnorm_shader, &ffn_norm_w.buf, &x_buf, &norm_x_buf, n_embd,
            );

            if layer.is_moe {
                self.dispatch_moe_ffn(
                    ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader, &activation_shader, &add_shader,
                    &norm_x_buf, &layer, layer_idx, &ffn_out_buf, n_embd,
                );
            } else {
                let ffn_gate_w = layer.dense.ffn_gate_w.as_ref().expect("ffn_gate_w");
                let ffn_up_w = layer.dense.ffn_up_w.as_ref().expect("ffn_up_w");
                let ffn_down_w = layer.dense.ffn_down_w.as_ref().expect("ffn_down_w");

                dispatch_proj_quant(
                    ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader, &norm_x_buf, &ffn_gate_w.buf, &ffn_gate_buf,
                    n_embd, self.model.ffn_length, ffn_gate_w.dtype,
                );
                dispatch_proj_quant(
                    ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader, &norm_x_buf, &ffn_up_w.buf, &ffn_up_buf,
                    n_embd, self.model.ffn_length, ffn_up_w.dtype,
                );
                dispatch_activation(
                    ctx, &activation_shader, &ffn_gate_buf, &ffn_up_buf,
                    &ffn_h_buf, self.model.ffn_length,
                );
                dispatch_proj_quant(
                    ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader, &ffn_h_buf, &ffn_down_w.buf, &ffn_out_buf,
                    self.model.ffn_length, n_embd, ffn_down_w.dtype,
                );
            }
            dispatch_add(ctx, &add_shader, &x_buf, &ffn_out_buf, &x_buf, n_embd);
        }

        if capture_set.contains(&self.layers.len()) {
            captured_states.push(x_buf.read_f32(n_embd));
        }

        if let Some(output_norm_w) = &self.output_norm_w {
            dispatch_rmsnorm(ctx, &rmsnorm_shader, &output_norm_w.buf, &x_buf, &x_buf, n_embd);
        }

        let final_hidden = x_buf.read_f32(n_embd);

        let logits = if let Some(output_w) = &self.output_weight {
            let vocab = self.vocab_size();
            let logits_buf = GpuBuffer::new(
                ctx, (vocab * 4) as u64, flags, mem_flags,
            );
            dispatch_proj_quant(
                ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader, &x_buf, &output_w.buf, &logits_buf, n_embd, vocab, output_w.dtype,
            );
            logits_buf.read_f32(vocab)
        } else {
            final_hidden.clone()
        };

        (logits, captured_states)
    }

    pub fn project_normalized_hidden(&self, hidden: &[f32]) -> Vec<f32> {
        let ctx = &self.ctx;
        let n_embd = self.embedding_length();
        assert_eq!(hidden.len(), n_embd);

        let flags = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mem_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let x_buf = GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags);
        let input_bytes: Vec<u8> = hidden.iter().flat_map(|v| v.to_le_bytes()).collect();
        x_buf.upload(&input_bytes);

        let output_w = self.output_weight.as_ref().expect("output_weight");
        let vocab = self.vocab_size();
        let logits_buf = GpuBuffer::new(ctx, (vocab * 4) as u64, flags, mem_flags);

        let proj_shader = ctx.create_shader_module(crate::include_spv!(proj));
        let proj_q8_0_shader = ctx.create_shader_module(crate::include_spv!(proj_q8_0));
        let proj_q4k_shader = ctx.create_shader_module(crate::include_spv!(proj_q4k));
        dispatch_proj_quant(
            ctx, &proj_q8_0_shader, &proj_q4k_shader, &proj_shader,
            &x_buf, &output_w.buf, &logits_buf, n_embd, vocab, output_w.dtype,
        );
        logits_buf.read_f32(vocab)
    }

    fn dispatch_moe_ffn(
        &self,
        ctx: &VulkanContext,
        quant_shader: &vk::ShaderModule,
        q4k_shader: &vk::ShaderModule,
        proj_shader: &vk::ShaderModule,
        activation_shader: &vk::ShaderModule,
        add_shader: &vk::ShaderModule,
        norm_x: &GpuBuffer,
        layer: &TargetLayer,
        layer_idx: usize,
        ffn_out_buf: &GpuBuffer,
        n_embd: usize,
    ) {
        let n_exp = self.model.n_expert;
        let n_exp_used = self.model.n_expert_used;
        let n_ff_exp = self.model.n_ff_exp;
        let n_shexp = self.model.shared_ffn_length;

        let moe = layer.moe.as_ref().expect("moe layer");

        let exp_probs_b_bias = moe.exp_probs_b_bias.as_ref().expect("exp_probs_b_bias");
        let ffn_gate_inp_w = moe.ffn_gate_inp_w.as_ref().expect("ffn_gate_inp_w");
        let ffn_gate_exps_w = moe.ffn_gate_exps_w.as_ref().expect("ffn_gate_exps_w");
        let ffn_up_exps_w = moe.ffn_up_exps_w.as_ref().expect("ffn_up_exps_w");
        let ffn_down_exps_w = moe.ffn_down_exps_w.as_ref().expect("ffn_down_exps_w");
        let ffn_gate_shexp_w = moe.ffn_gate_shexp_w.as_ref().expect("ffn_gate_shexp_w");
        let ffn_up_shexp_w = moe.ffn_up_shexp_w.as_ref().expect("ffn_up_shexp_w");
        let ffn_down_shexp_w = moe.ffn_down_shexp_w.as_ref().expect("ffn_down_shexp_w");

        let flags = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mem_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let gate_scores_buf = GpuBuffer::new(ctx, (n_exp * 4) as u64, flags, mem_flags);
        let moe_out_buf = GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags);
        let moe_expert_buf = GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags);
        let gate_buf = GpuBuffer::new(ctx, (n_ff_exp * 4) as u64, flags, mem_flags);
        let up_buf = GpuBuffer::new(ctx, (n_ff_exp * 4) as u64, flags, mem_flags);
        let hidden_buf = GpuBuffer::new(ctx, (n_ff_exp * 4) as u64, flags, mem_flags);

        let zero_embd: Vec<u8> = vec![0u8; n_embd * 4];

        dispatch_proj_quant(
            ctx, quant_shader, q4k_shader, proj_shader, norm_x, &ffn_gate_inp_w.buf, &gate_scores_buf,
            n_embd, n_exp, ffn_gate_inp_w.dtype,
        );

        let logits = gate_scores_buf.read_f32(n_exp);
        let probs: Vec<f32> = logits.iter().map(|&s| 1.0 / (1.0 + (-s).exp())).collect();
        let bias = exp_probs_b_bias.buf.read_f32(n_exp);
        let selection_probs: Vec<f32> = probs.iter().zip(bias.iter()).map(|(p, b)| p + b).collect();
        let mut indexed: Vec<(f32, usize)> = selection_probs.iter().copied().enumerate().map(|(i, s)| (s, i)).collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let selected_exp: Vec<usize> = indexed.iter().take(n_exp_used).map(|x| x.1).collect();
        let mut selected_scores: Vec<f32> = selected_exp.iter().map(|&i| probs[i]).collect();

        if self.model.reader.metadata.expert_weights_norm.unwrap_or(false) {
            let sum: f32 = selected_scores.iter().sum();
            let denom = sum.max(6.103515625e-5);
            for score in &mut selected_scores {
                *score /= denom;
            }
        }

        let weight_scale = self.model.reader.metadata.expert_weights_scale.unwrap_or(1.0);
        if weight_scale != 0.0 && weight_scale != 1.0 {
            for score in &mut selected_scores {
                *score *= weight_scale;
            }
        }

        moe_out_buf.upload(&zero_embd);

        let elem_stride = n_embd * n_ff_exp;
        for (&e, &sc) in selected_exp.iter().zip(selected_scores.iter()) {
            let off = (e * elem_stride) as u32;

            dispatch_proj_quant_expert(
                ctx, quant_shader, q4k_shader, proj_shader,
                norm_x, &ffn_gate_exps_w.buf, &gate_buf,
                n_embd, n_ff_exp, ffn_gate_exps_w.dtype, off,
            );
            dispatch_proj_quant_expert(
                ctx, quant_shader, q4k_shader, proj_shader,
                norm_x, &ffn_up_exps_w.buf, &up_buf,
                n_embd, n_ff_exp, ffn_up_exps_w.dtype, off,
            );
            dispatch_activation(
                ctx, activation_shader,
                &gate_buf, &up_buf, &hidden_buf, n_ff_exp,
            );
            let hidden = hidden_buf.read_f32(n_ff_exp);
            let col_l2 = hidden
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .clamp(1e-8, 1e30);
            let f16_safe = 32768.0f32;
            let hidden_scaled: Vec<u8> = hidden
                .iter()
                .flat_map(|v| (v * f16_safe / col_l2).to_le_bytes())
                .collect();
            hidden_buf.upload(&hidden_scaled);
            dispatch_proj_quant_expert(
                ctx, quant_shader, q4k_shader, proj_shader,
                &hidden_buf, &ffn_down_exps_w.buf, &moe_expert_buf,
                n_ff_exp, n_embd, ffn_down_exps_w.dtype, off,
            );
            if sc > 0.0 {
                let expert_out = moe_expert_buf.read_f32(n_embd);
                let out_scale = (col_l2 / f16_safe) * sc;
                let scaled: Vec<u8> = expert_out.iter().flat_map(|v| (v * out_scale).to_le_bytes()).collect();
                moe_expert_buf.upload(&scaled);
            }
            dispatch_add(ctx, add_shader, &moe_out_buf, &moe_expert_buf, &moe_out_buf, n_embd);
        }

        let shex_gate_buf = GpuBuffer::new(ctx, (n_shexp * 4) as u64, flags, mem_flags);
        let shex_up_buf = GpuBuffer::new(ctx, (n_shexp * 4) as u64, flags, mem_flags);
        let shex_h_buf = GpuBuffer::new(ctx, (n_shexp * 4) as u64, flags, mem_flags);
        let shex_out_buf = GpuBuffer::new(ctx, (n_embd * 4) as u64, flags, mem_flags);

        dispatch_proj_quant(
            ctx, quant_shader, q4k_shader, proj_shader, norm_x, &ffn_gate_shexp_w.buf, &shex_gate_buf,
            n_embd, n_shexp, ffn_gate_shexp_w.dtype,
        );
        dispatch_proj_quant(
            ctx, quant_shader, q4k_shader, proj_shader, norm_x, &ffn_up_shexp_w.buf, &shex_up_buf,
            n_embd, n_shexp, ffn_up_shexp_w.dtype,
        );
        dispatch_activation(
            ctx, activation_shader,
            &shex_gate_buf, &shex_up_buf, &shex_h_buf, n_shexp,
        );
        dispatch_proj_quant(
            ctx, quant_shader, q4k_shader, proj_shader, &shex_h_buf, &ffn_down_shexp_w.buf, &shex_out_buf,
            n_shexp, n_embd, ffn_down_shexp_w.dtype,
        );
        dispatch_add(ctx, add_shader, &moe_out_buf, &shex_out_buf, ffn_out_buf, n_embd);
    }
}

fn dispatch_mha_attn(
    ctx: &VulkanContext,
    attn_shader: &vk::ShaderModule,
    q_normed: &GpuBuffer,
    k_normed: &GpuBuffer,
    v_rep: &GpuBuffer,
    out: &GpuBuffer,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    dispatch_mha_attn_seq(ctx, attn_shader, q_normed, k_normed, v_rep, out, n_q_heads, n_kv_heads, head_dim, 1, 0);
}

#[allow(clippy::too_many_arguments)]
fn dispatch_mha_attn_seq(
    ctx: &VulkanContext,
    attn_shader: &vk::ShaderModule,
    q_normed: &GpuBuffer,
    k_normed: &GpuBuffer,
    v_rep: &GpuBuffer,
    out: &GpuBuffer,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    q_pos: usize,
) {
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let push: [u8; 32] = {
        let mut b = [0u8; 32];
        b[..4].copy_from_slice(&(n_q_heads as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(n_kv_heads as u32).to_le_bytes());
        b[8..12].copy_from_slice(&(head_dim as u32).to_le_bytes());
        b[12..16].copy_from_slice(&(seq_len as u32).to_le_bytes());
        b[16..20].copy_from_slice(&scale.to_le_bytes());
        b[20..24].copy_from_slice(&(q_pos as u32).to_le_bytes());
        b[24..28].copy_from_slice(&1u32.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(4);
    let pool = ctx.create_descriptor_pool(4);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, q_normed.buffer, q_normed.size);
    ctx.write_buffer_descriptor(set, 1, k_normed.buffer, k_normed.size);
    ctx.write_buffer_descriptor(set, 2, v_rep.buffer, v_rep.size);
    ctx.write_buffer_descriptor(set, 3, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 32);
    let pipeline = ctx.create_compute_pipeline(*attn_shader, layout);
    let groups = (n_q_heads * head_dim).div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

fn dispatch_moe_select(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    x: &GpuBuffer,
    router_w: &GpuBuffer,
    scores: &GpuBuffer,
    selected: &GpuBuffer,
    n_exp: usize,
    n_used: usize,
) {
    let push: [u8; 12] = {
        let mut b = [0u8; 12];
        b[..4].copy_from_slice(&(n_exp as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(n_used as u32).to_le_bytes());
        b[8..12].copy_from_slice(&0u32.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(4);
    let pool = ctx.create_descriptor_pool(4);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, router_w.buffer, router_w.size);
    ctx.write_buffer_descriptor(set, 2, scores.buffer, scores.size);
    ctx.write_buffer_descriptor(set, 3, selected.buffer, selected.size);

    let layout = ctx.create_pipeline_layout(set_layout, 12);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = 1u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_moe_ffn_kernel(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    x: &GpuBuffer,
    gate_exps_w: &GpuBuffer,
    up_exps_w: &GpuBuffer,
    down_exps_w: &GpuBuffer,
    selected: &GpuBuffer,
    out: &GpuBuffer,
    n_embd: usize,
    n_exp: usize,
    n_used: usize,
    n_ff_exp: usize,
) {
    let push: [u8; 24] = {
        let mut b = [0u8; 24];
        b[..4].copy_from_slice(&(n_embd as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(n_exp as u32).to_le_bytes());
        b[8..12].copy_from_slice(&(n_used as u32).to_le_bytes());
        b[12..16].copy_from_slice(&(n_ff_exp as u32).to_le_bytes());
        b[16..20].copy_from_slice(&0u32.to_le_bytes());
        b[20..24].copy_from_slice(&0u32.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(6);
    let pool = ctx.create_descriptor_pool(6);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, gate_exps_w.buffer, gate_exps_w.size);
    ctx.write_buffer_descriptor(set, 2, up_exps_w.buffer, up_exps_w.size);
    ctx.write_buffer_descriptor(set, 3, down_exps_w.buffer, down_exps_w.size);
    ctx.write_buffer_descriptor(set, 4, selected.buffer, selected.size);
    ctx.write_buffer_descriptor(set, 5, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 24);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n_embd.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_qk_norm_rope(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    x: &GpuBuffer,
    norm_w: &GpuBuffer,
    freq: &GpuBuffer,
    out: &GpuBuffer,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: u32,
) {
    let eps: f32 = 1e-6;
    let push: [u8; 20] = {
        let mut b = [0u8; 20];
        b[..4].copy_from_slice(&(n_heads as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(head_dim as u32).to_le_bytes());
        b[8..12].copy_from_slice(&(rope_dim as u32).to_le_bytes());
        b[12..16].copy_from_slice(&eps.to_le_bytes());
        b[16..20].copy_from_slice(&pos.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(4);
    let pool = ctx.create_descriptor_pool(4);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, norm_w.buffer, norm_w.size);
    ctx.write_buffer_descriptor(set, 2, freq.buffer, freq.size);
    ctx.write_buffer_descriptor(set, 3, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 20);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = (n_heads * head_dim).div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_qk_norm_rope_swa(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    x: &GpuBuffer,
    norm_w: &GpuBuffer,
    freq: &GpuBuffer,
    out: &GpuBuffer,
    n_heads: usize,
    head_dim: usize,
    rope_dim_swa: usize,
    pos: u32,
) {
    let eps: f32 = 1e-6;
    let push: [u8; 20] = {
        let mut b = [0u8; 20];
        b[..4].copy_from_slice(&(n_heads as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(head_dim as u32).to_le_bytes());
        b[8..12].copy_from_slice(&(rope_dim_swa as u32).to_le_bytes());
        b[12..16].copy_from_slice(&eps.to_le_bytes());
        b[16..20].copy_from_slice(&pos.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(4);
    let pool = ctx.create_descriptor_pool(4);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, norm_w.buffer, norm_w.size);
    ctx.write_buffer_descriptor(set, 2, freq.buffer, freq.size);
    ctx.write_buffer_descriptor(set, 3, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 20);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = (n_heads * head_dim).div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
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

fn dispatch_gate_attn(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    attn_out: &GpuBuffer,
    gate: &GpuBuffer,
    result: &GpuBuffer,
    n_heads: usize,
    head_dim: usize,
) {
    let push: [u8; 8] = {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&(n_heads as u32).to_le_bytes());
        b[4..].copy_from_slice(&(head_dim as u32).to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, attn_out.buffer, attn_out.size);
    ctx.write_buffer_descriptor(set, 1, gate.buffer, gate.size);
    ctx.write_buffer_descriptor(set, 2, result.buffer, result.size);

    let layout = ctx.create_pipeline_layout(set_layout, 8);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = (n_heads * head_dim).div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

fn dispatch_qk_norm(
    ctx: &VulkanContext,
    shader: &vk::ShaderModule,
    x: &GpuBuffer,
    w: &GpuBuffer,
    out: &GpuBuffer,
    n_heads: usize,
    n: usize,
) {
    let eps: f32 = 1e-6;
    let push: [u8; 12] = {
        let mut b = [0u8; 12];
        b[..4].copy_from_slice(&(n_heads as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        b[8..12].copy_from_slice(&eps.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, w.buffer, w.size);
    ctx.write_buffer_descriptor(set, 2, out.buffer, out.size);

    let layout = ctx.create_pipeline_layout(set_layout, 12);
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
    row_major: u32,
) {
    let push: [u8; 12] = {
        let mut b = [0u8; 12];
        b[..4].copy_from_slice(&(k as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        b[8..12].copy_from_slice(&row_major.to_le_bytes());
        b
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
    ctx.write_buffer_descriptor(set, 1, w.buffer, w.size);
    ctx.write_buffer_descriptor(set, 2, c.buffer, c.size);

    let layout = ctx.create_pipeline_layout(set_layout, 12);
    let pipeline = ctx.create_compute_pipeline(*shader, layout);
    let groups = n.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
}

fn dispatch_proj_quant(
    ctx: &VulkanContext,
    quant_shader: &vk::ShaderModule,
    q4k_shader: &vk::ShaderModule,
    f32_shader: &vk::ShaderModule,
    x: &GpuBuffer,
    w: &GpuBuffer,
    c: &GpuBuffer,
    k: usize,
    n: usize,
    dtype: WeightType,
) {
    dispatch_proj_quant_expert(ctx, quant_shader, q4k_shader, f32_shader, x, w, c, k, n, dtype, 0);
}

fn dispatch_proj_quant_expert(
    ctx: &VulkanContext,
    quant_shader: &vk::ShaderModule,
    q4k_shader: &vk::ShaderModule,
    f32_shader: &vk::ShaderModule,
    x: &GpuBuffer,
    w: &GpuBuffer,
    c: &GpuBuffer,
    k: usize,
    n: usize,
    dtype: WeightType,
    expert_offset: u32,
) {
    match dtype {
        WeightType::F32 => {
            dispatch_proj(ctx, f32_shader, x, w, c, k, n, 0);
        }
        WeightType::Q8_0 => {
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
            let pipeline = ctx.create_compute_pipeline(*quant_shader, layout);
            let groups = n.div_ceil(256) as u32;
            ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
        }
        WeightType::Q4_K => {
            let push: [u8; 12] = {
                let mut b = [0u8; 12];
                b[..4].copy_from_slice(&(k as u32).to_le_bytes());
                b[4..8].copy_from_slice(&(n as u32).to_le_bytes());
                b[8..12].copy_from_slice(&expert_offset.to_le_bytes());
                b
            };

            let set_layout = ctx.create_descriptor_set_layout(3);
            let pool = ctx.create_descriptor_pool(3);
            let set = ctx.allocate_descriptor_set(pool, set_layout);

            ctx.write_buffer_descriptor(set, 0, x.buffer, x.size);
            ctx.write_buffer_descriptor(set, 1, w.buffer, w.size);
            ctx.write_buffer_descriptor(set, 2, c.buffer, c.size);

            let layout = ctx.create_pipeline_layout(set_layout, 12);
            let pipeline = ctx.create_compute_pipeline(*q4k_shader, layout);
            let groups = n.div_ceil(256) as u32;
            ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));
        }
    }
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
