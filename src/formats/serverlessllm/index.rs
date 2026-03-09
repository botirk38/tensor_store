//! ServerlessLLM index parsing and metadata.

use crate::backends;
use crate::formats::error::{ReaderError, ReaderResult};
use crate::formats::traits::TensorMetadata;
use std::collections::HashMap;
use std::path::Path;

use super::tensor::IndexEntry;

/// Parsed ServerlessLLM index with tensor metadata (pure data, no state).
#[derive(Debug, Clone)]
pub struct Index {
    tensors: HashMap<String, IndexEntry>,
}

impl Index {
     /// Creates a new empty index.
     #[inline]
     #[must_use]
     pub fn new() -> Self {
         Self {
             tensors: HashMap::new(),
         }
     }

     /// Gets a tensor entry by name.
    #[inline]
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&IndexEntry> {
        self.tensors.get(name)
    }

    /// Returns the number of tensors tracked by this index.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true when the index has no tensors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns tensor names.
    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }

    /// Returns an iterator over tensor names and entries.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &IndexEntry)> {
        self.tensors.iter()
    }

    /// Returns the partition IDs used by this index.
    #[inline]
    #[must_use]
    pub fn partition_ids(&self) -> Vec<usize> {
        let mut ids: Vec<_> = self
            .tensors
            .values()
            .map(|entry| entry.partition_id)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort_unstable();
        ids
     }

     /// Parse index from raw bytes.
     pub fn from_bytes(data: &[u8]) -> ReaderResult<Self> {
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(data)
            .map_err(|err| ReaderError::ServerlessLlm(format!("JSON parse error: {err}")))?;

        let mut tensors = HashMap::with_capacity(raw.len());

        for (name, value) in raw {
            let arr = value.as_array().ok_or_else(|| {
                ReaderError::ServerlessLlm(format!("tensor entry '{name}' must be an array"))
            })?;

            // Support both 5-field (offset, size, shape, stride, dtype) and
            // 6-field entries where partition_id is present.
            let (offset, size, shape, stride, dtype, partition_id) = match arr.len() {
                5 => {
                    let offset = parse_u64(arr.first().expect("array has 5 elements"), &name)?;
                    let size = parse_u64(arr.get(1).expect("array has 5 elements"), &name)?;
                    let shape = parse_usize_vec(arr.get(2).expect("array has 5 elements"), &name)?;
                    let stride = parse_usize_vec(arr.get(3).expect("array has 5 elements"), &name)?;
                    let dtype = parse_string(arr.get(4).expect("array has 5 elements"), &name)?;
                    (offset, size, shape, stride, dtype, 0usize)
                }
                6 => {
                    let offset = parse_u64(arr.first().expect("array has 6 elements"), &name)?;
                    let size = parse_u64(arr.get(1).expect("array has 6 elements"), &name)?;
                    let shape = parse_usize_vec(arr.get(2).expect("array has 6 elements"), &name)?;
                    let stride = parse_usize_vec(arr.get(3).expect("array has 6 elements"), &name)?;
                    let dtype = parse_string(arr.get(4).expect("array has 6 elements"), &name)?;
                    let partition_id =
                        parse_usize(arr.get(5).expect("array has 6 elements"), &name)?;
                    (offset, size, shape, stride, dtype, partition_id)
                }
                _ => {
                    return Err(ReaderError::ServerlessLlm(format!(
                        "tensor entry '{}' must have 5 or 6 elements, got {}",
                        name,
                        arr.len()
                    )));
                }
            };

            tensors.insert(
                name,
                IndexEntry {
                    offset,
                    size,
                    shape,
                    stride,
                    dtype,
                    partition_id,
                },
            );
         }

         Ok(Index {
             tensors,
         })
     }

    /// Load index from file asynchronously.
    pub async fn load(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let data = backends::async_backend()
            .load(path.as_ref())
            .await?;
        Self::from_bytes(&data)
    }

    /// Load index from file synchronously.
    pub fn load_sync(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let data = std::fs::read(path.as_ref())?;
        Self::from_bytes(&data)
    }
}

impl<'a> IntoIterator for &'a Index {
    type Item = (&'a String, &'a IndexEntry);
    type IntoIter = std::collections::hash_map::Iter<'a, String, IndexEntry>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

impl TensorMetadata for Index {
    #[inline]
    fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.tensors
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }
}

impl Default for Index {
    fn default() -> Self {
        Self::new()
    }
}

// Helper parsers for JSON values
fn parse_u64(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<u64> {
    value.as_u64().ok_or_else(|| {
        ReaderError::ServerlessLlm(format!(
            "expected u64 in tensor '{tensor_name}', got {value:?}"
        ))
    })
}

fn parse_usize(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| {
            ReaderError::ServerlessLlm(format!(
                "expected usize in tensor '{tensor_name}', got {value:?}"
            ))
        })
}

fn parse_string(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<String> {
    value
        .as_str()
        .map(std::string::ToString::to_string)
        .ok_or_else(|| {
            ReaderError::ServerlessLlm(format!(
                "expected string in tensor '{tensor_name}', got {value:?}"
            ))
        })
}

fn parse_usize_vec(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<Vec<usize>> {
    let arr = value.as_array().ok_or_else(|| {
        ReaderError::ServerlessLlm(format!(
            "expected array in tensor '{tensor_name}', got {value:?}"
        ))
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(v.as_u64().ok_or_else(|| {
            ReaderError::ServerlessLlm(format!(
                "expected integer in tensor '{tensor_name}', got {v:?}"
            ))
        })? as usize);
    }
     Ok(out)
}

// ---------------------------------------------------------------------------
// Public API: Free Functions
// ---------------------------------------------------------------------------

/// Parse index from file path asynchronously.
pub async fn parse_index(path: impl AsRef<std::path::Path>) -> ReaderResult<Index> {
    Index::load(path).await
}

/// Parse index from file path synchronously.
pub fn parse_index_sync(path: impl AsRef<std::path::Path>) -> ReaderResult<Index> {
     Index::load_sync(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_index_with_5_fields() {
        let data = br#"{
            "tensor_a": [0, 4, [2,2], [2,1], "f32"]
        }"#;

        let index = Index::from_bytes(data).expect("parse index");
        let tensor_a = index.get("tensor_a").expect("tensor_a");
        assert_eq!(tensor_a.partition_id, 0);
        assert_eq!(tensor_a.size, 4);
        assert_eq!(tensor_a.shape, vec![2, 2]);
    }

    #[test]
    fn parse_index_with_6_fields() {
        let data = br#"{
            "tensor_b": [4, 2, [1,2], [2,1], "i8", 3]
        }"#;

        let index = Index::from_bytes(data).expect("parse index");
        let tensor_b = index.get("tensor_b").expect("tensor_b");
        assert_eq!(tensor_b.partition_id, 3);
        assert_eq!(tensor_b.size, 2);
        assert_eq!(tensor_b.shape, vec![1, 2]);
    }

    #[test]
    fn parse_index_rejects_invalid_length() {
        let data = br#"{"bad": [1,2,3,4]}"#;
        let err = Index::from_bytes(data).unwrap_err();
        assert!(
            format!("{err}").contains("must have 5 or 6 elements"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn tensor_names_works() {
        let data = br#"{"a": [0, 4, [2, 2], [2, 1], "f32"], "b": [4, 8, [2, 4], [4, 1], "f32"]}"#;
        let index = Index::from_bytes(data).expect("parse");
        let mut names = index.tensor_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn partition_ids_collects_and_sorts() {
        let data = br#"{"a": [0, 4, [2, 2], [2, 1], "f32", 2], "b": [4, 8, [2, 4], [4, 1], "f32", 0], "c": [12, 8, [2, 4], [4, 1], "f32", 2]}"#;
        let index = Index::from_bytes(data).expect("parse");
        assert_eq!(index.partition_ids(), vec![0, 2]);
    }
}
