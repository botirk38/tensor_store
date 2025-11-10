pub type IoResult<T> = std::io::Result<T>;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod backends;
pub mod converters;
pub mod readers;
pub mod writers;

use readers::safetensors::{OwnedSafeTensors, SafeTensorError};

pub async fn load_safetensors(path: &str) -> Result<OwnedSafeTensors, SafeTensorError> {
    readers::safetensors::load(path).await
}

pub async fn load_safetensors_parallel(path: &str) -> Result<OwnedSafeTensors, SafeTensorError> {
    readers::safetensors::load_parallel(path, 4).await
}

pub async fn load_safetensors_parallel_with_chunks(
    path: &str,
    chunks: usize,
) -> Result<OwnedSafeTensors, SafeTensorError> {
    readers::safetensors::load_parallel(path, chunks).await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "linux")]
    #[test]
    fn test_load_parallel_zero_copy() {
        // Test with the existing test file
        let path = "test_model.safetensors";

        tokio_uring::start(async {
            // Load with parallel loading (zero-copy)
            let data = load_safetensors_parallel(path).await.unwrap();

            // Verify we can deserialize it
            let tensors = data.tensors();
            assert!(!tensors.names().is_empty());

            // Load with single-threaded loading for comparison
            let data_single = load_safetensors(path).await.unwrap();

            // Data should be identical
            assert_eq!(data.as_bytes().len(), data_single.as_bytes().len());
            assert_eq!(data.as_bytes(), data_single.as_bytes());

            println!(
                "Zero-copy parallel loading test passed! Loaded {} tensors",
                tensors.names().len()
            );
        });
    }

    #[cfg(not(target_os = "linux"))]
    #[tokio::test]
    async fn test_load_parallel_zero_copy() {
        // Test with the existing test file
        let path = "test_model.safetensors";

        // Load with parallel loading (zero-copy)
        let data = load_safetensors_parallel(path).await.unwrap();

        // Verify we can deserialize it
        let tensors = data.tensors();
        assert!(!tensors.names().is_empty());

        // Load with single-threaded loading for comparison
        let data_single = load_safetensors(path).await.unwrap();

        // Data should be identical
        assert_eq!(data.as_bytes().len(), data_single.as_bytes().len());
        assert_eq!(data.as_bytes(), data_single.as_bytes());

        println!(
            "Zero-copy parallel loading test passed! Loaded {} tensors",
            tensors.names().len()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_load_parallel_zero_copy_io_uring() {
        // Test io_uring version specifically
        let path = "test_model.safetensors";

        tokio_uring::start(async {
            // Load with parallel loading (zero-copy)
            let data = load_safetensors_parallel(path).await.unwrap();

            // Verify we can deserialize it
            let tensors = data.tensors();
            assert!(!tensors.names().is_empty());

            // Load with single-threaded loading for comparison
            let data_single = load_safetensors(path).await.unwrap();

            // Data should be identical
            assert_eq!(data.as_bytes().len(), data_single.as_bytes().len());
            assert_eq!(data.as_bytes(), data_single.as_bytes());

            println!(
                "Zero-copy parallel loading (io_uring) test passed! Loaded {} tensors",
                tensors.names().len()
            );
        });
    }
}
