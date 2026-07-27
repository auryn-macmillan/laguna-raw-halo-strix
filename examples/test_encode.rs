use laguna_raw::dflash::DFlashModel;
use std::path::Path;

fn main() {
    let model = DFlashModel::load(Path::new("/home/auryn/models/laguna/laguna-s-2.1-DFlash-BF16.gguf")).expect("load");
    let n_embd_inp = model.n_embd_inp();
    
    let features: Vec<f32> = (0..n_embd_inp).map(|i| {
        (i as f32 * 0.01).sin() * 0.1
    }).collect();
    
    let enc_out = model.encode(&features);
    println!("encoder output (first 10): {:?}", &enc_out[..10]);
    println!("all zeros? {}", enc_out.iter().all(|v| *v == 0.0));
    println!("range: [{}, {}]", 
        enc_out.iter().cloned().fold(f32::INFINITY, f32::min),
        enc_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    let logits = model.forward_encode_and_decode(&features);
    println!("logits (first 10): {:?}", &logits[..10]);
    println!("all zeros? {}", logits.iter().all(|v| *v == 0.0));
    println!("has NaN? {}", logits.iter().any(|v: &f32| v.is_nan()));
}
