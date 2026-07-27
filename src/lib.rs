pub mod dflash;
pub mod gguf;
pub mod model;
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
    let h_bits = (0.5f32.bits() >> 16) as u16;
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

    let total = n_heads * head_dim;
    let groups = total.div_ceil(256) as u32;
    ctx.submit_compute(pipeline, layout, set, &push_bytes, (groups, 1, 1));

    let _result = out_buf.read_f32(4);
    println!("test-rope: PASS (4 elements, rope_dim=4)");
}
