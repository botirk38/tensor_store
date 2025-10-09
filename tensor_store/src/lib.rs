use std::io::Result as IoResult;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod loaders;


#[cfg(target_os = "linux")]
pub async fn load_safetensors(path: &str) -> IoResult<Vec<u8>> {
    loaders::uring::BasicLoader::load(path).await
}



#[cfg(not(target_os = "linux"))]
pub async fn load_safetensors(path: &str) -> IoResult<Vec<u8>> {
    loaders::tokio::BasicLoader::load(path).await
}

