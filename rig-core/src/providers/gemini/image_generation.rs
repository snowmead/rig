// ================================================================
//! Google Gemini Image Generation Integration
//! From [Gemini API Reference](https://ai.google.dev/gemini-api/docs/image-generation)
// ================================================================

use super::completion::gemini_api_types::{
    Content, GenerateContentRequest, GenerateContentResponse, GenerationConfig, ImageConfig,
    PartKind, ResponseModality, Role,
};
use super::Client;
use crate::http_client::HttpClientExt;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::image_generation;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use serde::Deserialize;

// ================================================================
// Gemini Image Generation Models
// ================================================================

/// Gemini 2.5 Flash with image generation capability (stable)
/// This is the recommended model for image generation.
pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";

/// Gemini 3 Pro with image generation capability (preview)
pub const GEMINI_3_PRO_IMAGE_PREVIEW: &str = "gemini-3-pro-image-preview";

/// Gemini 2.5 Flash image preview model
/// Note: This model will be retired on October 31, 2025. Migrate to GEMINI_2_5_FLASH_IMAGE.
#[deprecated(note = "Will be retired on October 31, 2025. Use GEMINI_2_5_FLASH_IMAGE instead")]
pub const GEMINI_2_5_FLASH_IMAGE_PREVIEW: &str = "gemini-2.5-flash-preview-image-generation";

/// Gemini 2.0 Flash experimental model with image generation
/// Note: This model will be retired on October 31, 2025. Migrate to GEMINI_2_5_FLASH_IMAGE.
#[deprecated(note = "Will be retired on October 31, 2025. Use GEMINI_2_5_FLASH_IMAGE instead")]
pub const GEMINI_2_0_FLASH_IMAGE_GENERATION: &str = "gemini-2.0-flash-preview-image-generation";

// ================================================================
// Image Generation Response Types
// ================================================================

/// Response from Gemini image generation API
#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
    /// The raw response from the Gemini API
    pub raw_response: GenerateContentResponse,
    /// The generated image as base64 encoded bytes
    pub image_data: Vec<u8>,
    /// The MIME type of the generated image
    pub mime_type: String,
    /// Any accompanying text from the model
    pub text: Option<String>,
}

impl TryFrom<GenerateContentResponse>
    for image_generation::ImageGenerationResponse<ImageGenerationResponse>
{
    type Error = ImageGenerationError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        let candidate = response.candidates.first().ok_or_else(|| {
            ImageGenerationError::ResponseError("No candidates in response".to_string())
        })?;

        let content = candidate.content.as_ref().ok_or_else(|| {
            ImageGenerationError::ResponseError("No content in response candidate".to_string())
        })?;

        let mut image_data: Option<Vec<u8>> = None;
        let mut mime_type = String::from("image/png");
        let mut text: Option<String> = None;

        for part in &content.parts {
            match &part.part {
                PartKind::InlineData(blob) => {
                    mime_type = blob.mime_type.clone();
                    image_data = Some(BASE64_STANDARD.decode(&blob.data).map_err(|e| {
                        ImageGenerationError::ResponseError(format!(
                            "Failed to decode base64 image: {}",
                            e
                        ))
                    })?);
                }
                PartKind::Text(t) => {
                    text = Some(t.clone());
                }
                _ => {}
            }
        }

        let image_bytes = image_data.ok_or_else(|| {
            ImageGenerationError::ResponseError(
                "No image data found in response. Make sure you're using an image-capable model like gemini-2.5-flash-preview-image-generation".to_string(),
            )
        })?;

        Ok(image_generation::ImageGenerationResponse {
            image: image_bytes.clone(),
            response: ImageGenerationResponse {
                raw_response: response,
                image_data: image_bytes,
                mime_type,
                text,
            },
        })
    }
}

// ================================================================
// Image Generation Model
// ================================================================

/// Gemini image generation model
#[derive(Clone, Debug)]
pub struct ImageGenerationModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    /// Name of the model (e.g.: gemini-2.5-flash-preview-image-generation)
    pub model: String,
}

impl<T> ImageGenerationModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Send + 'static,
{
    type Response = ImageGenerationResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
    {
        // Build the generation config with image modalities
        let generation_config = GenerationConfig {
            response_modalities: Some(vec![ResponseModality::Text, ResponseModality::Image]),
            image_config: Some(ImageConfig {
                aspect_ratio: calculate_aspect_ratio(request.width, request.height),
                image_size: None,
                number_of_images: Some(1),
            }),
            // Disable defaults that might interfere with image generation
            temperature: None,
            max_output_tokens: None,
            ..Default::default()
        };

        let gemini_request = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![request.prompt.into()],
                role: Some(Role::User),
            }],
            generation_config: Some(generation_config),
            safety_settings: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
            additional_params: request.additional_params,
        };

        tracing::trace!(
            target: "rig::image_generation",
            "Sending image generation request to Gemini API: {}",
            serde_json::to_string_pretty(&gemini_request).unwrap_or_default()
        );

        let body = serde_json::to_vec(&gemini_request)?;
        let path = format!("/v1beta/models/{}:generateContent", self.model);

        let http_request = self
            .client
            .post(path.as_str())?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

        let response = self.client.send::<_, Vec<u8>>(http_request).await?;

        if response.status().is_success() {
            let response_body = response
                .into_body()
                .await
                .map_err(ImageGenerationError::HttpError)?;

            let response_text = String::from_utf8_lossy(&response_body).to_string();
            tracing::debug!(
                target: "rig::image_generation",
                "Received raw response from Gemini API: {}",
                response_text
            );

            let gemini_response: GenerateContentResponse =
                serde_json::from_slice(&response_body).map_err(|err| {
                    tracing::error!(
                        error = %err,
                        body = %response_text,
                        "Failed to deserialize Gemini image generation response"
                    );
                    ImageGenerationError::JsonError(err)
                })?;

            gemini_response.try_into()
        } else {
            let status = response.status();
            let body = response
                .into_body()
                .await
                .map_err(ImageGenerationError::HttpError)?;
            let text = String::from_utf8_lossy(&body);

            Err(ImageGenerationError::ProviderError(format!(
                "{}: {}",
                status, text
            )))
        }
    }
}

/// Calculate aspect ratio string from width and height
fn calculate_aspect_ratio(width: u32, height: u32) -> Option<String> {
    // Find GCD to simplify the ratio
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }

    let divisor = gcd(width, height);
    let w = width / divisor;
    let h = height / divisor;

    // Map to supported Gemini aspect ratios
    // Supported: 1:1, 3:2, 2:3, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
    let supported_ratios = [
        (1, 1),
        (3, 2),
        (2, 3),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 4),
        (9, 16),
        (16, 9),
        (21, 9),
    ];

    // Find the closest supported ratio
    let ratio = (w as f64) / (h as f64);
    let closest = supported_ratios
        .iter()
        .min_by(|a, b| {
            let a_ratio = (a.0 as f64) / (a.1 as f64);
            let b_ratio = (b.0 as f64) / (b.1 as f64);
            (a_ratio - ratio)
                .abs()
                .partial_cmp(&(b_ratio - ratio).abs())
                .unwrap()
        })
        .unwrap();

    Some(format!("{}:{}", closest.0, closest.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_aspect_ratio() {
        // Square
        assert_eq!(calculate_aspect_ratio(1024, 1024), Some("1:1".to_string()));

        // 16:9 landscape
        assert_eq!(calculate_aspect_ratio(1920, 1080), Some("16:9".to_string()));

        // 9:16 portrait
        assert_eq!(calculate_aspect_ratio(1080, 1920), Some("9:16".to_string()));

        // 4:3
        assert_eq!(calculate_aspect_ratio(1600, 1200), Some("4:3".to_string()));

        // 3:4
        assert_eq!(calculate_aspect_ratio(1200, 1600), Some("3:4".to_string()));

        // Approximate ratio should map to closest
        assert_eq!(calculate_aspect_ratio(1000, 1000), Some("1:1".to_string()));
    }
}
