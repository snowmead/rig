use rig::image_generation::ImageGenerationModel;
use rig::prelude::*;
use rig::providers::gemini;
use std::env::args;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_PATH: &str = "./gemini_output.png";

#[tokio::main]
async fn main() {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let arguments: Vec<String> = args().collect();

    let path = if arguments.len() > 1 {
        arguments[1].clone()
    } else {
        DEFAULT_PATH.to_string()
    };

    let path = Path::new(&path);
    let mut file = File::create(path).expect("Failed to create file");

    // Create Gemini client from environment variable GEMINI_API_KEY
    let client = gemini::Client::from_env();

    // Create an image generation model using Gemini 2.5 Flash Image
    let model = client.image_generation_model(gemini::GEMINI_2_5_FLASH_IMAGE);

    println!("Generating image with Gemini...");

    let response = model
        .image_generation_request()
        .prompt("A majestic castle sitting upon a large mountain, overlooking a serene lake at sunset, painted in a watercolor style")
        .width(1024)
        .height(1024)
        .send()
        .await
        .expect("Failed to generate image");

    // Write the image to file
    file.write_all(&response.image)
        .expect("Failed to write image to file");

    println!("Image saved to: {}", path.display());

    // Print any accompanying text from the model
    if let Some(text) = &response.response.text {
        println!("Model response: {}", text);
    }

    println!("Image MIME type: {}", response.response.mime_type);
}
