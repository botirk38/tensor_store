use super::IoResult;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncReadExt;

pub struct BasicLoader;

impl BasicLoader {
    pub async fn load(path: &str) -> IoResult<Vec<u8>> {
        let mut file = TokioFile::open(path).await?;
        let metadata = file.metadata().await?;
        let mut buf = vec![0u8; metadata.len() as usize];
        file.read_exact(&mut buf).await?;
        
        super::validate_safetensors(&buf)?;
        Ok(buf)
    }
}
