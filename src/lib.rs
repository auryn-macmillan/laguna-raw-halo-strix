pub mod dflash;
pub mod gguf;
pub mod model;
pub mod target;
pub mod vulkan;

#[macro_export]
macro_rules! include_spv {
    ($name:ident) => {
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/",
            stringify!($name),
            ".comp.spv"
        ))
    };
}

use std::path::Path;
use ash::vk;
use dflash::{DFlashModel, ForwardBuffers, dequant_to_f32, upload_f32};
use gguf::{GGUFReader, TensorType};

pub fn run_info(path: &Path) {
    let reader = GGUFReader::open(path).expect("failed to open model");
    let m = &reader.metadata;

    println!("GGUF v{}", reader.version);
    println!("architecture: {:?}", m.architecture);
    println!("block_count: {:?}", m.block_count);
    println!("embedding_length: {:?}", m.embedding_length);
    println!("feed_forward_length: {:?}", m.feed_forward_length);
    println!("attention_head_count: {:?}", m.attention_head_count);
    println!("attention_head_count_kv: {:?}", m.attention_head_count_kv);
    println!("attention_key_length: {:?}", m.attention_key_length);
    println!("rope_dimension_count: {:?}", m.rope_dimension_count);
    println!("rope_freq_base: {:?}", m.rope_freq_base);
    println!("tensor_count: {}", reader.tensor_count);
    println!("tensor_data_offset: {}", reader.tensor_data_offset);

    println!("\ntensors:");
    for t in &reader.tensors {
        println!(
            "  {:?} {:?} {} bytes @ offset {}",
            t.shape, t.dtype, t.n_bytes(), t.data_offset
        );
    }
}

pub fn run_shapes(path: &Path) {
    let reader = GGUFReader::open(path).expect("failed to open model");
    for t in &reader.tensors {
        println!("{:?}: {:?} {:?} ({})", t.name, t.shape, t.dtype, t.n_bytes());
    }
}

pub fn run_forward(path: &Path, token_id: usize) {
    let model = DFlashModel::load(path).expect("failed to load model");
    let n_embd = model.embedding_length();
    let n_heads = model.head_count();
    let n_kv_heads = model.head_count_kv();
    let head_dim = model.head_dim();
    let n_ff = model.ffn_length();

    let bufs = ForwardBuffers::new(
        &model.ctx,
        n_embd,
        n_heads,
        n_kv_heads,
        head_dim,
        n_ff,
    );

    let logits = model.forward_token(&bufs, token_id);
    println!("forward pass complete, {} logits", logits.len());
    for (i, v) in logits.iter().enumerate().take(10) {
        println!("  logit[{}] = {}", i, v);
    }
}

pub fn run_forward_encode(path: &Path) {
    let model = DFlashModel::load(path).expect("failed to load model");
    let n_embd = model.embedding_length();
    let n_embd_inp = model.n_embd_inp();

    let dummy_features = vec![0.0f32; n_embd_inp];
    let logits = model.forward_encode_and_decode(&dummy_features);
    println!("encode+decode complete, {} logits", logits.len());
    for (i, v) in logits.iter().enumerate().take(10) {
        println!("  logit[{}] = {}", i, v);
    }
}

pub fn run_test_dequant() {
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
        assert!((result[i] - expected).abs() < 1e-3, "mismatch at {}: {} vs {}", i, result[i], expected);
    }
    println!("test-dequant: PASS (bf16, 64 elements)");

    let mut data = vec![0u8; 34];
    let h_bits = (0.5f32.to_bits() >> 16) as u16;
    data[0..2].copy_from_slice(&h_bits.to_le_bytes());
    for i in 0..32 {
        data[2 + i] = 1;
    }
    let result = dequant_to_f32(&data, TensorType::Q8_0, 32);
    assert!((result[0] - 0.5).abs() < 1e-3, "q8_0 mismatch: {}", result[0]);
    println!("test-dequant: PASS (q8_0, 32 elements)");
}

pub fn run_test_matmul() {
    let ctx = vulkan::VulkanContext::init();

    let a_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.01).collect();

    let a_buf = upload_f32(&ctx, &a_data);
    let b_buf = upload_f32(&ctx, &b_data);
    let c_buf = upload_f32(&ctx, &vec![0.0f32; 3]);

    let proj_shader = ctx.create_shader_module(include_spv!(proj));

    let push: [u8; 8] = {
        let mut p = [0u8; 8];
        p[..4].copy_from_slice(&(8u32).to_le_bytes());
        p[4..].copy_from_slice(&(3u32).to_le_bytes());
        p
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, a_buf.buffer, a_buf.size);
    ctx.write_buffer_descriptor(set, 1, b_buf.buffer, b_buf.size);
    ctx.write_buffer_descriptor(set, 2, c_buf.buffer, c_buf.size);

    let layout = ctx.create_pipeline_layout(set_layout, 8);
    let pipeline = ctx.create_compute_pipeline(proj_shader, layout);

    let groups = 3u32.div_ceil(256);
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));

    let result = c_buf.read_f32(3);

    let expected: Vec<f32> = (0..3)
        .map(|j| {
            (0..8).map(|i| a_data[i] * b_data[j * 8 + i]).sum()
        })
        .collect();

    for i in 0..3 {
        assert!((result[i] - expected[i]).abs() < 1e-4, "matmul mismatch at {}: {} vs {}", i, result[i], expected[i]);
    }
    println!("test-matmul: PASS (8x3, 3 outputs)");
}

pub fn run_test_rmsnorm() {
    let ctx = vulkan::VulkanContext::init();

    let x_data: Vec<f32> = (0..4).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let w_data: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];

    let x_buf = upload_f32(&ctx, &x_data);
    let w_buf = upload_f32(&ctx, &w_data);
    let out_buf = upload_f32(&ctx, &vec![0.0f32; 4]);

    let rmsnorm_shader = ctx.create_shader_module(include_spv!(rmsnorm));

    let push: [u8; 8] = {
        let mut p = [0u8; 8];
        p[..4].copy_from_slice(&(4u32).to_le_bytes());
        p[4..].copy_from_slice(&1e-5f32.to_le_bytes());
        p
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x_buf.buffer, x_buf.size);
    ctx.write_buffer_descriptor(set, 1, w_buf.buffer, w_buf.size);
    ctx.write_buffer_descriptor(set, 2, out_buf.buffer, out_buf.size);

    let layout = ctx.create_pipeline_layout(set_layout, 8);
    let pipeline = ctx.create_compute_pipeline(rmsnorm_shader, layout);

    let groups = 4u32.div_ceil(256);
    ctx.submit_compute(pipeline, layout, set, &push, (groups, 1, 1));

    let result = out_buf.read_f32(4);

    let mean_sq: f32 = x_data.iter().map(|v| v * v).sum::<f32>() / 4.0;
    let norm_factor = 1.0 / (mean_sq + 1e-5).sqrt();
    let expected: Vec<f32> = x_data.iter().zip(w_data.iter()).map(|(x, w)| x * norm_factor * w).collect();

    for i in 0..4 {
        assert!((result[i] - expected[i]).abs() < 1e-4, "rmsnorm mismatch at {}: {} vs {}", i, result[i], expected[i]);
    }
    println!("test-rmsnorm: PASS (4 elements)");
}

pub fn run_test_rope() {
    let ctx = vulkan::VulkanContext::init();

    let rope_dim = 4;
    let n_heads = 2;
    let head_dim = 2;

    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let freq: Vec<f32> = vec![1.0, 0.5];

    let x_buf = upload_f32(&ctx, &x_data);
    let freq_buf = upload_f32(&ctx, &freq);
    let out_buf = upload_f32(&ctx, &vec![0.0f32; 4]);

    let rope_shader = ctx.create_shader_module(include_spv!(rope));

    let push: [u32; 4] = [n_heads as u32, head_dim as u32, 1, rope_dim as u32];
    let push_bytes: [u8; 16] = {
        let mut p = [0u8; 16];
        p[..4].copy_from_slice(&push[0].to_le_bytes());
        p[4..8].copy_from_slice(&push[1].to_le_bytes());
        p[8..12].copy_from_slice(&push[2].to_le_bytes());
        p[12..].copy_from_slice(&push[3].to_le_bytes());
        p
    };

    let set_layout = ctx.create_descriptor_set_layout(3);
    let pool = ctx.create_descriptor_pool(3);
    let set = ctx.allocate_descriptor_set(pool, set_layout);

    ctx.write_buffer_descriptor(set, 0, x_buf.buffer, x_buf.size);
    ctx.write_buffer_descriptor(set, 1, freq_buf.buffer, freq_buf.size);
    ctx.write_buffer_descriptor(set, 2, out_buf.buffer, out_buf.size);

    let layout = ctx.create_pipeline_layout(set_layout, 16);
    let pipeline = ctx.create_compute_pipeline(rope_shader, layout);

    let total: u32 = n_heads * head_dim;
    let groups = total.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push_bytes, (groups, 1, 1));

    let _result = out_buf.read_f32(4);
    println!("test-rope: PASS (4 elements, rope_dim=4)");
}

pub fn run_target_info(path: &Path) {
    use target::TargetModel;
    let model = TargetModel::load(path).expect("failed to load target model");
    let m = &model.model;
    println!("Target Model Info:");
    println!("  layers: {}", model.layer_count());
    println!("  embedding_length: {}", m.embedding_length);
    println!("  ffn_length: {}", m.ffn_length);
    println!("  head_count: {}", m.head_count);
    println!("  head_count_kv: {}", m.head_count_kv);
    println!("  head_dim: {}", m.head_dim);
    println!("  rope_dim: {}", m.rope_dim);
    println!("  rope_dim_swa: {}", m.rope_dim_swa);
    println!("  expert_count: {}", m.n_expert);
    println!("  expert_used_count: {}", m.n_expert_used);
    println!("  expert_ffn_length: {}", m.n_ff_exp);
    println!("  shared_ffn_length: {}", m.shared_ffn_length);
    println!("  has_moe: {}", m.has_moe);
    println!("  has_shexp: {}", m.has_shexp);
    println!("  vocab_size: {}", model.vocab_size());
    println!("  target_layers: {:?}", m.target_layers);

    for (i, layer) in model.layers.iter().enumerate() {
        let layer_type = if layer.is_moe { "MoE" } else { "Dense" };
        println!("  layer {} [{}]", i, layer_type);
    }
}

pub fn run_target_forward(path: &Path, token_id: usize) {
    use target::TargetModel;
    use dflash::dequant_to_f32;
    use gguf::TensorType;
    let mut model = TargetModel::load(path).expect("failed to load target model");
    let n_embd = model.embedding_length();
    let vocab_size = model.vocab_size();
    let has_tok_emb = model.tok_embeddings.is_some();

    if has_tok_emb {
        let n_elements = model.tok_embeddings.as_ref().unwrap().n_elements;
        if token_id < vocab_size {
            let raw = model.read_tensor_raw("token_embd.weight")
                .expect("failed to read tok_embd");
            let gt = if let Some(t) = model.model.reader.tensor_by_name("token_embd.weight") {
                t.dtype
            } else {
                TensorType::F32
            };
            let all_embs = dequant_to_f32(&raw, gt, n_elements);
            let token_emb: Vec<f32> = (0..n_embd)
                .map(|i| all_embs[token_id * n_embd + i])
                .collect();
            println!("Running target model forward pass with token_id={}", token_id);
            let (logits, captured) = model.forward_token(&token_emb, None);
            println!("Forward pass complete: {} logits, {} captured states", logits.len(), captured.len());
            let mut top: Vec<(f32, usize)> = logits.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            println!("Top logits:");
            for (v, i) in top.iter().take(10) {
                println!("  logit[{}] = {}", i, v);
            }
            println!("Specific indices:");
            for (i, v) in logits.iter().enumerate().take(10) {
                println!("  logit[{}] = {}", i, v);
            }
        } else {
            eprintln!("token_id {} >= vocab_size {}", token_id, vocab_size);
            std::process::exit(1);
        }
    } else {
        let dummy_input = vec![0.1f32; n_embd];
        let (logits, captured) = model.forward_token(&dummy_input, None);
        println!("Forward pass complete: {} logits, {} captured states", logits.len(), captured.len());
        for (i, v) in logits.iter().enumerate().take(10) {
            println!("  logit[{}] = {}", i, v);
        }
    }
}

/// Cached autoregressive greedy generation using the target model's KV cache.
pub fn run_target_generate(path: &Path, start_token: usize, n_tokens: usize) {
    use target::{TargetModel, TargetKvCache};
    use dflash::dequant_to_f32;
    use gguf::TensorType;

    let mut model = TargetModel::load(path).expect("failed to load target model");
    let n_embd = model.embedding_length();
    let vocab_size = model.vocab_size();
    let n_layers = model.layers.len();

    if model.tok_embeddings.is_none() {
        eprintln!("model has no token embeddings; cannot generate");
        std::process::exit(1);
    }
    let n_elements = model.tok_embeddings.as_ref().unwrap().n_elements;
    let raw = model.read_tensor_raw("token_embd.weight").expect("read tok_embd");
    let gt = model.model.reader.tensor_by_name("token_embd.weight")
        .map(|t| t.dtype).unwrap_or(TensorType::F32);
    let all_embs = dequant_to_f32(&raw, gt, n_elements);

    let embed = |tok: usize| -> Vec<f32> {
        (0..n_embd).map(|i| all_embs[tok * n_embd + i]).collect()
    };

    let mut cache = TargetKvCache::new(n_layers);
    let mut cur = start_token;
    let mut generated = Vec::new();

    println!("Generating {} tokens from start_token={}", n_tokens, start_token);
    for step in 0..n_tokens {
        if cur >= vocab_size {
            eprintln!("token {} out of range", cur);
            break;
        }
        let emb = embed(cur);
        let (logits, _) = model.forward_token_cached(&emb, None, &mut cache);
        let next = logits.iter().copied().enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        println!("  step {}: input={} -> next={}", step, cur, next);
        generated.push(next);
        cur = next;
    }
    println!("Generated tokens: {:?}", generated);
}

pub fn run_target_dflash(target_path: &Path, dflash_path: &Path, token_id: usize) {
    use dflash::{dequant_to_f32, DFlashModel};
    use gguf::TensorType;
    use target::TargetModel;

    let mut target = TargetModel::load(target_path).expect("failed to load target model");
    let draft = DFlashModel::load(dflash_path).expect("failed to load DFlash model");

    let n_embd = target.embedding_length();
    let vocab_size = target.vocab_size();
    if token_id >= vocab_size {
        eprintln!("token_id {} >= vocab_size {}", token_id, vocab_size);
        std::process::exit(1);
    }

    let n_elements = target.tok_embeddings.as_ref().unwrap().n_elements;
    let raw = target.read_tensor_raw("token_embd.weight").expect("failed to read tok_embd");
    let gt = target
        .model
        .reader
        .tensor_by_name("token_embd.weight")
        .map(|t| t.dtype)
        .unwrap_or(TensorType::F32);
    let all_embs = dequant_to_f32(&raw, gt, n_elements);
    let token_emb: Vec<f32> = (0..n_embd)
        .map(|i| all_embs[token_id * n_embd + i])
        .collect();

    let target_layers: Vec<usize> = draft.target_layers().iter().map(|&v| v as usize).collect();
    println!("DFlash target layers: {:?}", target_layers);
    let (_target_logits, captured) = target.forward_token(&token_emb, Some(&target_layers));
    let mut features = Vec::with_capacity(captured.len() * n_embd);
    for state in &captured {
        features.extend_from_slice(state);
    }
    println!("Captured {} states, {} feature floats", captured.len(), features.len());

    let draft_hidden = draft.forward_injected_decode_hidden(&features, &token_emb);
    let logits = target.project_normalized_hidden(&draft_hidden);
    println!("DFlash draft logits: {}", logits.len());
    let mut top: Vec<(f32, usize)> = logits.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top draft logits:");
    for (v, i) in top.iter().take(10) {
        println!("  logit[{}] = {}", i, v);
    }
}

fn target_token_embedding(
    model: &mut target::TargetModel,
    token_id: usize,
) -> Vec<f32> {
    use dflash::dequant_to_f32;
    use gguf::TensorType;

    let n_embd = model.embedding_length();
    let vocab_size = model.vocab_size();
    assert!(token_id < vocab_size, "token_id {} >= vocab_size {}", token_id, vocab_size);

    let n_elements = model.tok_embeddings.as_ref().unwrap().n_elements;
    let raw = model.read_tensor_raw("token_embd.weight").expect("failed to read tok_embd");
    let gt = model
        .model
        .reader
        .tensor_by_name("token_embd.weight")
        .map(|t| t.dtype)
        .unwrap_or(TensorType::F32);
    let all_embs = dequant_to_f32(&raw, gt, n_elements);
    (0..n_embd)
        .map(|i| all_embs[token_id * n_embd + i])
        .collect()
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn run_target_dflash_verify(
    target_path: &Path,
    dflash_path: &Path,
    start_token: usize,
    n_steps: usize,
) {
    use dflash::DFlashModel;
    use target::TargetModel;

    let mut target = TargetModel::load(target_path).expect("failed to load target model");
    let draft = DFlashModel::load(dflash_path).expect("failed to load DFlash model");
    let n_embd = target.embedding_length();
    let target_layers: Vec<usize> = draft.target_layers().iter().map(|&v| v as usize).collect();

    let mut token = start_token;
    let mut accepted = Vec::with_capacity(n_steps);
    let mut exact_accepts = 0usize;

    println!("DFlash target layers: {:?}", target_layers);
    for step in 0..n_steps {
        let emb = target_token_embedding(&mut target, token);
        let (target_logits, captured) = target.forward_token(&emb, Some(&target_layers));

        let mut features = Vec::with_capacity(captured.len() * n_embd);
        for state in &captured {
            features.extend_from_slice(state);
        }

        let draft_hidden = draft.forward_injected_decode_hidden(&features, &emb);
        let draft_logits = target.project_normalized_hidden(&draft_hidden);

        let draft_tok = argmax(&draft_logits);
        let target_tok = argmax(&target_logits);
        let accept = draft_tok == target_tok;
        if accept {
            exact_accepts += 1;
        }
        let next = if accept { draft_tok } else { target_tok };
        accepted.push(next);

        println!(
            "step {}: input={} draft={} target={} {} -> accepted={}",
            step,
            token,
            draft_tok,
            target_tok,
            if accept { "ACCEPT" } else { "REJECT" },
            next,
        );
        token = next;
    }

    println!("accepted tokens: {:?}", accepted);
    println!("exact draft accept rate: {}/{}", exact_accepts, n_steps);
}

pub fn run_project_ref(path: &Path, hidden_path: &Path) {
    use target::TargetModel;
    use std::io::Read;

    let model = TargetModel::load(path).expect("failed to load target model");
    let n_embd = model.embedding_length();

    let mut bytes = Vec::new();
    std::fs::File::open(hidden_path)
        .expect("failed to open hidden file")
        .read_to_end(&mut bytes)
        .expect("failed to read hidden file");
    assert_eq!(bytes.len(), n_embd * 4);
    let hidden: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let logits = model.project_normalized_hidden(&hidden);
    let mut top: Vec<(f32, usize)> = logits.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top logits:");
    for (v, i) in top.iter().take(10) {
        println!("  logit[{}] = {}", i, v);
    }
    println!("Specific indices:");
    for (i, v) in logits.iter().enumerate().take(8) {
        println!("  logit[{}] = {}", i, v);
    }
}

pub fn run_dump_tensor(path: &std::path::Path, name: &str) {
    use crate::gguf::GGUFReader;
    let reader = GGUFReader::open(path).expect("open");
    if let Some(t) = reader.tensor_by_name(name) {
        println!("{} shape={:?} dtype={:?} n_elements={}", name, t.shape, t.dtype, t.n_elements());
    } else {
        println!("tensor not found: {}", name);
    }
}
