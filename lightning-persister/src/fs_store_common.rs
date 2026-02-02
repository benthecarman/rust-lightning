//! Common utilities shared between [`FilesystemStore`] and [`FilesystemStoreV2`].
//!
//! [`FilesystemStore`]: crate::fs_store::FilesystemStore
//! [`FilesystemStoreV2`]: crate::fs_store_v2::FilesystemStoreV2

use crate::utils::{check_namespace_key_validity, is_valid_kvstore_str};

use lightning::types::string::PrintableString;
use lightning::util::persist::{PageToken, PaginatedListResponse};

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::UNIX_EPOCH;

#[cfg(target_os = "windows")]
use std::ffi::OsStr;
#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStrExt;

/// Calls a Windows API function and returns Ok(()) on success or the last OS error on failure.
#[cfg(target_os = "windows")]
macro_rules! call {
	($e: expr) => {
		if $e != 0 {
			Ok(())
		} else {
			Err(std::io::Error::last_os_error())
		}
	};
}

#[cfg(target_os = "windows")]
use call;

/// Converts a path to a null-terminated wide string for Windows API calls.
#[cfg(target_os = "windows")]
fn path_to_windows_str<T: AsRef<OsStr>>(path: &T) -> Vec<u16> {
	path.as_ref().encode_wide().chain(Some(0)).collect()
}

/// The directory name used for empty namespaces in v2.
/// Uses brackets which are not in KVSTORE_NAMESPACE_KEY_ALPHABET, preventing collisions
/// with valid namespace names.
pub(crate) const EMPTY_NAMESPACE_DIR: &str = "[empty]";

/// Inner state shared between sync and async operations for filesystem stores.
///
/// This struct manages the data directory, temporary file counter, and per-path locks
/// that ensure we don't have concurrent writes to the same file.
pub(crate) struct FilesystemStoreState {
	pub(crate) data_dir: PathBuf,
	tmp_file_counter: AtomicUsize,
	/// Per path lock that ensures that we don't have concurrent writes to the same file.
	/// The lock also encapsulates the latest written version per key.
	locks: Mutex<HashMap<PathBuf, Arc<RwLock<u64>>>>,
	/// Version counter to ensure that writes are applied in the correct order.
	/// It is assumed that read and list operations aren't sensitive to the order of execution.
	next_version: AtomicU64,
}

impl FilesystemStoreState {
	/// Creates a new `FilesystemStoreState` with the given data directory.
	pub(crate) fn new(data_dir: PathBuf) -> Self {
		Self {
			data_dir,
			tmp_file_counter: AtomicUsize::new(0),
			locks: Mutex::new(HashMap::new()),
			next_version: AtomicU64::new(1),
		}
	}

	/// Gets or creates a lock reference for the given path.
	fn get_inner_lock_ref(&self, path: PathBuf) -> Arc<RwLock<u64>> {
		let mut outer_lock = self.locks.lock().unwrap();
		Arc::clone(&outer_lock.entry(path).or_default())
	}

	/// Cleans up unused locks to prevent memory leaks.
	///
	/// If there are no arcs in use elsewhere (besides the map entry and the provided reference),
	/// we can remove the map entry to prevent leaking memory.
	fn clean_locks(&self, inner_lock_ref: &Arc<RwLock<u64>>, dest_file_path: PathBuf) {
		let mut outer_lock = self.locks.lock().unwrap();

		let strong_count = Arc::strong_count(inner_lock_ref);
		debug_assert!(strong_count >= 2, "Unexpected FilesystemStore strong count");

		if strong_count == 2 {
			outer_lock.remove(&dest_file_path);
		}
	}

	/// Executes a read operation while holding the read lock for the given path.
	fn execute_locked_read<F: FnOnce() -> Result<(), lightning::io::Error>>(
		&self, dest_file_path: PathBuf, callback: F,
	) -> Result<(), lightning::io::Error> {
		let inner_lock_ref = self.get_inner_lock_ref(dest_file_path.clone());
		let res = {
			let _guard = inner_lock_ref.read().unwrap();
			callback()
		};
		self.clean_locks(&inner_lock_ref, dest_file_path);
		res
	}

	/// Executes a write operation with version tracking.
	///
	/// Returns `Ok(true)` if the callback was executed, `Ok(false)` if skipped due to staleness.
	fn execute_locked_write<F: FnOnce() -> Result<(), lightning::io::Error>>(
		&self, inner_lock_ref: Arc<RwLock<u64>>, lock_key: PathBuf, version: u64, callback: F,
	) -> Result<bool, lightning::io::Error> {
		let res = {
			let mut last_written_version = inner_lock_ref.write().unwrap();

			// Check if we already have a newer version written/removed. This is used in async
			// contexts to realize eventual consistency.
			let is_stale_version = version <= *last_written_version;

			// If the version is not stale, we execute the callback. Otherwise we can and must skip.
			if is_stale_version {
				Ok(false)
			} else {
				callback().map(|_| {
					*last_written_version = version;
					true
				})
			}
		};

		self.clean_locks(&inner_lock_ref, lock_key);

		res
	}

	/// Returns the number of in-flight locks (for testing).
	#[cfg(any(all(feature = "tokio", test), fuzzing))]
	pub(crate) fn state_size(&self) -> usize {
		self.locks.lock().unwrap().len()
	}

	/// Returns the base directory path for a namespace combination.
	///
	/// On Windows, this canonicalizes the path after creating the data directory.
	fn get_base_dir_path(&self) -> std::io::Result<PathBuf> {
		#[cfg(target_os = "windows")]
		{
			let data_dir = self.data_dir.clone();
			fs::create_dir_all(data_dir.clone())?;
			fs::canonicalize(data_dir)
		}
		#[cfg(not(target_os = "windows"))]
		{
			Ok(self.data_dir.clone())
		}
	}

	/// Generates a unique temporary file path based on the destination path.
	fn get_tmp_file_path(&self, dest_file_path: &PathBuf) -> PathBuf {
		let mut tmp_file_path = dest_file_path.clone();
		let tmp_file_ext = format!("{}.tmp", self.tmp_file_counter.fetch_add(1, Ordering::AcqRel));
		tmp_file_path.set_extension(tmp_file_ext);
		tmp_file_path
	}

	/// Generates a unique trash file path for Windows deletion operations.
	#[cfg(target_os = "windows")]
	fn get_trash_file_path(&self, dest_file_path: &PathBuf) -> PathBuf {
		let mut trash_file_path = dest_file_path.clone();
		let trash_file_ext =
			format!("{}.trash", self.tmp_file_counter.fetch_add(1, Ordering::AcqRel));
		trash_file_path.set_extension(trash_file_ext);
		trash_file_path
	}

	/// Validates namespaces/key and resolves the filesystem path.
	///
	/// When `use_empty_ns_dir` is true (v2), empty namespaces are replaced with [`EMPTY_NAMESPACE_DIR`]
	/// to give a consistent two-level directory structure. When false (v1), empty namespaces are
	/// simply omitted from the path, giving variable depth.
	pub(crate) fn resolve_path(
		&self, primary_namespace: &str, secondary_namespace: &str, key: Option<&str>,
		operation: &str, use_empty_ns_dir: bool,
	) -> lightning::io::Result<PathBuf> {
		check_namespace_key_validity(primary_namespace, secondary_namespace, key, operation)?;

		let mut path = self.get_base_dir_path()?;

		if use_empty_ns_dir {
			path.push(if primary_namespace.is_empty() {
				EMPTY_NAMESPACE_DIR
			} else {
				primary_namespace
			});
			path.push(if secondary_namespace.is_empty() {
				EMPTY_NAMESPACE_DIR
			} else {
				secondary_namespace
			});
		} else {
			path.push(primary_namespace);
			if !secondary_namespace.is_empty() {
				path.push(secondary_namespace);
			}
		}

		if let Some(key) = key {
			path.push(key);
		}

		Ok(path)
	}

	/// Writes data to a temporary file and prepares it for atomic rename.
	///
	/// This handles:
	/// - Creating the parent directory
	/// - Writing to a temporary file
	/// - Setting mtime if requested (for FilesystemStoreV2)
	/// - Syncing the temp file
	///
	/// Returns the temporary file path that should be renamed to the destination.
	fn prepare_atomic_write(
		&self, dest_file_path: &PathBuf, buf: &[u8], preserve_mtime: Option<std::time::SystemTime>,
	) -> lightning::io::Result<PathBuf> {
		let parent_directory = dest_file_path.parent().ok_or_else(|| {
			let msg =
				format!("Could not retrieve parent directory of {}.", dest_file_path.display());
			std::io::Error::new(std::io::ErrorKind::InvalidInput, msg)
		})?;
		fs::create_dir_all(parent_directory)?;

		let tmp_file_path = self.get_tmp_file_path(dest_file_path);

		{
			let tmp_file = fs::File::create(&tmp_file_path)?;
			let mut writer = std::io::BufWriter::new(&tmp_file);
			writer.write_all(buf)?;
			writer.flush()?;

			// If we need to preserve the original mtime (for updates), set it before fsync.
			if let Some(mtime) = preserve_mtime {
				let times = std::fs::FileTimes::new().set_modified(mtime);
				tmp_file.set_times(times)?;
			}

			tmp_file.sync_all()?;
		}

		Ok(tmp_file_path)
	}

	/// Removes a file on Windows using the trash file approach for durability.
	#[cfg(target_os = "windows")]
	fn remove_file_windows(&self, dest_file_path: &PathBuf) -> lightning::io::Result<()> {
		// Since Windows `DeleteFile` API is not persisted until the last open file handle
		// is dropped, and there seemingly is no reliable way to flush the directory
		// metadata, we here fall back to use a 'recycling bin' model, i.e., first move the
		// file to be deleted to a temporary trash file and remove the latter file
		// afterwards.
		//
		// This should be marginally better, as, according to the documentation,
		// `MoveFileExW` APIs should offer stronger persistence guarantees,
		// at least if `MOVEFILE_WRITE_THROUGH`/`MOVEFILE_REPLACE_EXISTING` is set.
		// However, all this is partially based on assumptions and local experiments, as
		// Windows API is horribly underdocumented.
		let trash_file_path = self.get_trash_file_path(dest_file_path);

		call!(unsafe {
			windows_sys::Win32::Storage::FileSystem::MoveFileExW(
				path_to_windows_str(&dest_file_path).as_ptr(),
				path_to_windows_str(&trash_file_path).as_ptr(),
				windows_sys::Win32::Storage::FileSystem::MOVEFILE_WRITE_THROUGH
					| windows_sys::Win32::Storage::FileSystem::MOVEFILE_REPLACE_EXISTING,
			)
		})?;

		{
			// We fsync the trash file in hopes this will also flush the original's file
			// metadata to disk.
			let trash_file =
				fs::OpenOptions::new().read(true).write(true).open(&trash_file_path)?;
			trash_file.sync_all()?;
		}

		// We're fine if this remove would fail as the trash file will be cleaned up in
		// list eventually.
		fs::remove_file(trash_file_path).ok();

		Ok(())
	}

	/// Gets a new version number and lock reference for the given path.
	fn get_new_version_and_lock_ref(&self, dest_file_path: PathBuf) -> (Arc<RwLock<u64>>, u64) {
		let version = self.next_version.fetch_add(1, Ordering::Relaxed);
		if version == u64::MAX {
			panic!("FilesystemStore version counter overflowed");
		}

		// Get a reference to the inner lock. We do this early so that the arc can double as an
		// in-flight counter for cleaning up unused locks.
		let inner_lock_ref = self.get_inner_lock_ref(dest_file_path);

		(inner_lock_ref, version)
	}

	/// Reads the contents of a file while holding the appropriate read lock.
	fn locked_read(&self, dest_file_path: PathBuf) -> lightning::io::Result<Vec<u8>> {
		let mut buf = Vec::new();

		self.execute_locked_read(dest_file_path.clone(), || {
			let mut f = fs::File::open(&dest_file_path)?;
			f.read_to_end(&mut buf)?;
			Ok(())
		})?;

		Ok(buf)
	}

	/// Writes a specific version of a key to the filesystem. If a newer version has been written
	/// already, this function returns early without writing.
	///
	/// When `preserve_mtime` is true, the file's existing modification time is preserved on update
	/// (used by FilesystemStoreV2 to maintain creation order for pagination).
	fn write_version(
		&self, inner_lock_ref: Arc<RwLock<u64>>, dest_file_path: PathBuf, buf: &[u8],
		preserve_mtime: bool, version: u64,
	) -> lightning::io::Result<bool> {
		let mtime = if preserve_mtime {
			fs::metadata(&dest_file_path).ok().and_then(|m| m.modified().ok())
		} else {
			None
		};

		let tmp_file_path = self.prepare_atomic_write(&dest_file_path, buf, mtime)?;

		self.execute_locked_write(inner_lock_ref, dest_file_path.clone(), version, || {
			#[cfg(not(target_os = "windows"))]
			{
				finalize_atomic_write_unix(&tmp_file_path, &dest_file_path)
			}

			#[cfg(target_os = "windows")]
			{
				finalize_atomic_write_windows(&tmp_file_path, &dest_file_path, mtime)
			}
		})
	}

	/// Removes a specific version of a key from the filesystem. If a newer version has been
	/// written already, this function returns early without removing.
	fn remove_version(
		&self, inner_lock_ref: Arc<RwLock<u64>>, dest_file_path: PathBuf, lazy: bool, version: u64,
	) -> lightning::io::Result<bool> {
		self.execute_locked_write(inner_lock_ref, dest_file_path.clone(), version, || {
			if !dest_file_path.is_file() {
				return Ok(());
			}

			if lazy {
				// If we're lazy we just call remove and be done with it.
				fs::remove_file(&dest_file_path)?;
			} else {
				// If we're not lazy we try our best to persist the updated metadata to ensure
				// atomicity of this call.
				#[cfg(not(target_os = "windows"))]
				{
					remove_file_unix(&dest_file_path)?;
				}

				#[cfg(target_os = "windows")]
				{
					self.remove_file_windows(&dest_file_path)?;
				}
			}

			Ok(())
		})
	}

	/// Lists all (primary_namespace, secondary_namespace, key) tuples in the store.
	///
	/// In v1 (`use_empty_ns_dir = false`), the directory structure has variable depth: files can
	/// appear at any level (representing keys with empty namespaces). In v2 (`use_empty_ns_dir =
	/// true`), the structure is always two directories deep, with [`EMPTY_NAMESPACE_DIR`] standing
	/// in for empty namespaces.
	pub(crate) fn list_all_keys_impl(
		&self, use_empty_ns_dir: bool,
	) -> lightning::io::Result<Vec<(String, String, String)>> {
		if !self.data_dir.exists() {
			return Ok(Vec::new());
		}

		let mut keys = Vec::new();

		'primary_loop: for primary_entry in fs::read_dir(&self.data_dir)? {
			let primary_entry = primary_entry?;
			let primary_path = primary_entry.path();

			// In v1 (variable depth), files at the root are keys with empty namespaces
			if !use_empty_ns_dir {
				if let Ok(Some(key)) = entry_to_key(&primary_entry, true) {
					keys.push((String::new(), String::new(), key));
					continue 'primary_loop;
				}
			}

			if !primary_path.is_dir() {
				continue 'primary_loop;
			}

			let primary_namespace = match dir_to_namespace(&primary_path, use_empty_ns_dir) {
				Some(ns) => ns,
				None => continue 'primary_loop,
			};

			'secondary_loop: for secondary_entry in fs::read_dir(&primary_path)? {
				let secondary_entry = secondary_entry?;
				let secondary_path = secondary_entry.path();

				// In v1, files at this level are keys with empty secondary namespace
				if !use_empty_ns_dir {
					if let Ok(Some(key)) = entry_to_key(&secondary_entry, true) {
						keys.push((primary_namespace.clone(), String::new(), key));
						continue 'secondary_loop;
					}
				}

				if !secondary_path.is_dir() {
					continue 'secondary_loop;
				}

				let secondary_namespace = match dir_to_namespace(&secondary_path, use_empty_ns_dir)
				{
					Some(ns) => ns,
					None => continue 'secondary_loop,
				};

				for key_entry in fs::read_dir(&secondary_path)? {
					let key_entry = key_entry?;
					if let Some(key) = entry_to_key(&key_entry, !use_empty_ns_dir)? {
						keys.push((primary_namespace.clone(), secondary_namespace.clone(), key));
					}
				}
			}
		}

		Ok(keys)
	}

	/// Sync entry point for reading a key from the filesystem.
	pub(crate) fn read_impl(
		&self, primary_namespace: &str, secondary_namespace: &str, key: &str,
		use_empty_ns_dir: bool,
	) -> lightning::io::Result<Vec<u8>> {
		let path = self.resolve_path(
			primary_namespace,
			secondary_namespace,
			Some(key),
			"read",
			use_empty_ns_dir,
		)?;
		self.locked_read(path)
	}

	/// Sync entry point for writing a key to the filesystem.
	pub(crate) fn write_impl(
		&self, primary_namespace: &str, secondary_namespace: &str, key: &str, buf: &[u8],
		preserve_mtime: bool, use_empty_ns_dir: bool,
	) -> lightning::io::Result<()> {
		let path = self.resolve_path(
			primary_namespace,
			secondary_namespace,
			Some(key),
			"write",
			use_empty_ns_dir,
		)?;
		let (inner_lock_ref, version) = self.get_new_version_and_lock_ref(path.clone());
		self.write_version(inner_lock_ref, path, buf, preserve_mtime, version).map(|_| ())
	}

	/// Sync entry point for removing a key from the filesystem.
	pub(crate) fn remove_impl(
		&self, primary_namespace: &str, secondary_namespace: &str, key: &str, lazy: bool,
		use_empty_ns_dir: bool,
	) -> lightning::io::Result<()> {
		let path = self.resolve_path(
			primary_namespace,
			secondary_namespace,
			Some(key),
			"remove",
			use_empty_ns_dir,
		)?;
		let (inner_lock_ref, version) = self.get_new_version_and_lock_ref(path.clone());
		self.remove_version(inner_lock_ref, path, lazy, version).map(|_| ())
	}

	/// Sync entry point for listing keys in a namespace.
	pub(crate) fn list_impl(
		&self, primary_namespace: &str, secondary_namespace: &str, use_empty_ns_dir: bool,
		retry_on_race: bool,
	) -> lightning::io::Result<Vec<String>> {
		let path = self.resolve_path(
			primary_namespace,
			secondary_namespace,
			None,
			"list",
			use_empty_ns_dir,
		)?;
		do_list(&path, retry_on_race)
	}

	/// Sync entry point for paginated listing.
	///
	/// Collects all entries with their modification times, sorts by mtime descending
	/// (newest first), and returns a page of results with an optional next page token.
	pub(crate) fn list_paginated_impl(
		&self, primary_namespace: &str, secondary_namespace: &str, page_token: Option<PageToken>,
	) -> lightning::io::Result<PaginatedListResponse> {
		let prefixed_dest = self.resolve_path(
			primary_namespace,
			secondary_namespace,
			None,
			"list_paginated",
			true,
		)?;
		if !prefixed_dest.exists() {
			return Ok(PaginatedListResponse { keys: Vec::new(), next_page_token: None });
		}

		// Collect all entries with their modification times
		let mut entries: Vec<(u64, String)> = Vec::new();
		for dir_entry in fs::read_dir(prefixed_dest)? {
			let dir_entry = dir_entry?;

			if let Some(key) = entry_to_key(&dir_entry, false)? {
				// Get modification time as millis since epoch
				let mtime_millis = dir_entry
					.metadata()
					.ok()
					.and_then(|m| m.modified().ok())
					.and_then(|t| t.duration_since(UNIX_EPOCH).ok())
					.map(|d| d.as_millis() as u64)
					.unwrap_or(0);

				entries.push((mtime_millis, key));
			}
		}

		// Sort by mtime descending (newest first), then by key descending for same mtime
		entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

		// Find starting position based on page token
		let start_idx = if let Some(token) = page_token {
			let (token_mtime, token_key) = parse_page_token(&token.0)?;

			// Find entries that come after the token (older entries = lower mtime)
			// or same mtime but lexicographically smaller key (since we sort descending)
			entries
				.iter()
				.position(|(mtime, key)| {
					*mtime < token_mtime
						|| (*mtime == token_mtime && key.as_str() < token_key.as_str())
				})
				.unwrap_or(entries.len())
		} else {
			0
		};

		// Take PAGE_SIZE entries starting from start_idx
		let page_entries: Vec<_> =
			entries.iter().skip(start_idx).take(PAGE_SIZE).cloned().collect();

		let keys: Vec<String> = page_entries.iter().map(|(_, key)| key.clone()).collect();

		// Determine next page token
		let next_page_token = if start_idx + PAGE_SIZE < entries.len() {
			page_entries.last().map(|(mtime, key)| PageToken(format_page_token(*mtime, key)))
		} else {
			None
		};

		Ok(PaginatedListResponse { keys, next_page_token })
	}

	/// Extracts version and lock from a path result for use in async write/remove operations.
	///
	/// The version counter is incremented synchronously (before the async block) so that ordering
	/// is established at call time, not execution time.
	#[cfg(feature = "tokio")]
	fn get_version_and_lock(
		&self, path_result: &lightning::io::Result<PathBuf>,
	) -> Option<((Arc<RwLock<u64>>, u64), PathBuf)> {
		path_result
			.as_ref()
			.ok()
			.map(|path| (self.get_new_version_and_lock_ref(path.clone()), path.clone()))
	}

	/// Async wrapper for [`Self::read_impl`] that runs the read on a blocking thread.
	#[cfg(feature = "tokio")]
	pub(crate) fn read_async(
		self: Arc<Self>, primary_namespace: &str, secondary_namespace: &str, key: &str,
		use_empty_ns_dir: bool,
	) -> impl core::future::Future<Output = Result<Vec<u8>, lightning::io::Error>> + Send {
		let path = self.resolve_path(
			primary_namespace,
			secondary_namespace,
			Some(key),
			"read",
			use_empty_ns_dir,
		);
		async move {
			let path = path?;
			tokio::task::spawn_blocking(move || self.locked_read(path)).await.unwrap_or_else(|e| {
				Err(lightning::io::Error::new(lightning::io::ErrorKind::Other, e))
			})
		}
	}

	/// Async wrapper for [`Self::write_version`] that runs the write on a blocking thread.
	///
	/// The version counter is incremented synchronously (before the returned future is polled) so
	/// that ordering is established at call time, not execution time.
	#[cfg(feature = "tokio")]
	pub(crate) fn write_async(
		self: Arc<Self>, primary_namespace: &str, secondary_namespace: &str, key: &str,
		buf: Vec<u8>, preserve_mtime: bool, use_empty_ns_dir: bool,
	) -> impl core::future::Future<Output = Result<(), lightning::io::Error>> + Send {
		let path_result = self.resolve_path(
			primary_namespace,
			secondary_namespace,
			Some(key),
			"write",
			use_empty_ns_dir,
		);
		let version_and_lock = self.get_version_and_lock(&path_result);
		async move {
			let ((inner_lock_ref, version), dest_file_path) = match version_and_lock {
				Some(v) => v,
				None => return Err(path_result.unwrap_err()),
			};

			tokio::task::spawn_blocking(move || {
				self.write_version(inner_lock_ref, dest_file_path, &buf, preserve_mtime, version)
					.map(|_| ())
			})
			.await
			.unwrap_or_else(|e| Err(lightning::io::Error::new(lightning::io::ErrorKind::Other, e)))
		}
	}

	/// Async wrapper for [`Self::remove_version`] that runs the remove on a blocking thread.
	///
	/// The version counter is incremented synchronously (before the returned future is polled) so
	/// that ordering is established at call time, not execution time.
	#[cfg(feature = "tokio")]
	pub(crate) fn remove_async(
		self: Arc<Self>, primary_namespace: &str, secondary_namespace: &str, key: &str, lazy: bool,
		use_empty_ns_dir: bool,
	) -> impl core::future::Future<Output = Result<(), lightning::io::Error>> + Send {
		let path_result = self.resolve_path(
			primary_namespace,
			secondary_namespace,
			Some(key),
			"remove",
			use_empty_ns_dir,
		);
		let version_and_lock = self.get_version_and_lock(&path_result);
		async move {
			let ((inner_lock_ref, version), dest_file_path) = match version_and_lock {
				Some(v) => v,
				None => return Err(path_result.unwrap_err()),
			};

			tokio::task::spawn_blocking(move || {
				self.remove_version(inner_lock_ref, dest_file_path, lazy, version).map(|_| ())
			})
			.await
			.unwrap_or_else(|e| Err(lightning::io::Error::new(lightning::io::ErrorKind::Other, e)))
		}
	}

	/// Async wrapper for [`Self::list_impl`] that runs the listing on a blocking thread.
	#[cfg(feature = "tokio")]
	pub(crate) fn list_async(
		&self, primary_namespace: &str, secondary_namespace: &str, use_empty_ns_dir: bool,
		retry_on_race: bool,
	) -> impl core::future::Future<Output = Result<Vec<String>, lightning::io::Error>> + Send {
		let path = self.resolve_path(
			primary_namespace,
			secondary_namespace,
			None,
			"list",
			use_empty_ns_dir,
		);
		async move {
			let path = path?;
			tokio::task::spawn_blocking(move || do_list(&path, retry_on_race)).await.unwrap_or_else(
				|e| Err(lightning::io::Error::new(lightning::io::ErrorKind::Other, e)),
			)
		}
	}
}

/// The fixed page size for paginated listing operations.
pub(crate) const PAGE_SIZE: usize = 50;

/// The length of the timestamp in a page token (milliseconds since epoch as 16-digit decimal).
const PAGE_TOKEN_TIMESTAMP_LEN: usize = 16;

/// Formats a page token from mtime (millis since epoch) and key.
pub(crate) fn format_page_token(mtime_millis: u64, key: &str) -> String {
	format!("{:016}:{}", mtime_millis, key)
}

/// Parses a page token into mtime (millis since epoch) and key.
pub(crate) fn parse_page_token(token: &str) -> lightning::io::Result<(u64, String)> {
	let colon_pos = token.find(':').ok_or_else(|| {
		lightning::io::Error::new(
			lightning::io::ErrorKind::InvalidInput,
			"Invalid page token format",
		)
	})?;

	if colon_pos != PAGE_TOKEN_TIMESTAMP_LEN {
		return Err(lightning::io::Error::new(
			lightning::io::ErrorKind::InvalidInput,
			"Invalid page token format",
		));
	}

	let mtime = token[..colon_pos].parse::<u64>().map_err(|_| {
		lightning::io::Error::new(
			lightning::io::ErrorKind::InvalidInput,
			"Invalid page token timestamp",
		)
	})?;

	let key = token[colon_pos + 1..].to_string();

	Ok((mtime, key))
}

/// Performs the atomic rename from temp file to destination on Unix.
#[cfg(not(target_os = "windows"))]
fn finalize_atomic_write_unix(
	tmp_file_path: &PathBuf, dest_file_path: &PathBuf,
) -> lightning::io::Result<()> {
	fs::rename(tmp_file_path, dest_file_path)?;

	let parent_directory = dest_file_path.parent().ok_or_else(|| {
		let msg = format!("Could not retrieve parent directory of {}.", dest_file_path.display());
		std::io::Error::new(std::io::ErrorKind::InvalidInput, msg)
	})?;

	let dir_file = fs::OpenOptions::new().read(true).open(parent_directory)?;
	dir_file.sync_all()?;
	Ok(())
}

/// Performs the atomic rename from temp file to destination on Windows.
#[cfg(target_os = "windows")]
fn finalize_atomic_write_windows(
	tmp_file_path: &PathBuf, dest_file_path: &PathBuf,
	preserve_mtime: Option<std::time::SystemTime>,
) -> lightning::io::Result<()> {
	let res = if dest_file_path.exists() {
		call!(unsafe {
			windows_sys::Win32::Storage::FileSystem::ReplaceFileW(
				path_to_windows_str(&dest_file_path).as_ptr(),
				path_to_windows_str(&tmp_file_path).as_ptr(),
				std::ptr::null(),
				windows_sys::Win32::Storage::FileSystem::REPLACEFILE_IGNORE_MERGE_ERRORS,
				std::ptr::null_mut() as *const core::ffi::c_void,
				std::ptr::null_mut() as *const core::ffi::c_void,
			)
		})
	} else {
		call!(unsafe {
			windows_sys::Win32::Storage::FileSystem::MoveFileExW(
				path_to_windows_str(&tmp_file_path).as_ptr(),
				path_to_windows_str(&dest_file_path).as_ptr(),
				windows_sys::Win32::Storage::FileSystem::MOVEFILE_WRITE_THROUGH
					| windows_sys::Win32::Storage::FileSystem::MOVEFILE_REPLACE_EXISTING,
			)
		})
	};

	match res {
		Ok(()) => {
			// Open the destination file to fsync it and set mtime if needed.
			let dest_file = fs::OpenOptions::new().read(true).write(true).open(dest_file_path)?;

			// On Windows, ReplaceFileW/MoveFileExW may not preserve the mtime we set
			// on the tmp file, so we explicitly set it again here.
			if let Some(mtime) = preserve_mtime {
				let times = std::fs::FileTimes::new().set_modified(mtime);
				dest_file.set_times(times)?;
			}

			dest_file.sync_all()?;
			Ok(())
		},
		Err(e) => Err(e.into()),
	}
}

/// Removes a file atomically on Unix with fsync on the parent directory.
#[cfg(not(target_os = "windows"))]
fn remove_file_unix(dest_file_path: &PathBuf) -> lightning::io::Result<()> {
	fs::remove_file(dest_file_path)?;

	let parent_directory = dest_file_path.parent().ok_or_else(|| {
		let msg = format!("Could not retrieve parent directory of {}.", dest_file_path.display());
		std::io::Error::new(std::io::ErrorKind::InvalidInput, msg)
	})?;
	let dir_file = fs::OpenOptions::new().read(true).open(parent_directory)?;
	// The above call to `fs::remove_file` corresponds to POSIX `unlink`, whose changes
	// to the inode might get cached (and hence possibly lost on crash), depending on
	// the target platform and file system.
	//
	// In order to assert we permanently removed the file in question we therefore
	// call `fsync` on the parent directory on platforms that support it.
	dir_file.sync_all()?;
	Ok(())
}

// The number of times we retry listing keys in `list` before we give up reaching
// a consistent view and error out.
const LIST_DIR_CONSISTENCY_RETRIES: usize = 10;

/// Checks whether a directory entry is a valid key file and returns its key name.
///
/// Returns `Ok(Some(key))` for valid key files, `Ok(None)` for entries that should be
/// skipped (temp files, trash files, invalid names), and `Err` if metadata cannot be read.
///
/// When `allow_dirs` is true (used by FilesystemStore v1), directories are silently skipped
/// since they represent namespace subdirectories. When false (used by FilesystemStoreV2),
/// directories are unexpected at the key level and produce an error.
pub(crate) fn entry_to_key(
	dir_entry: &fs::DirEntry, allow_dirs: bool,
) -> Result<Option<String>, lightning::io::Error> {
	let p = dir_entry.path();

	if let Some(ext) = p.extension() {
		#[cfg(target_os = "windows")]
		{
			// Clean up any trash files lying around.
			if ext == "trash" {
				fs::remove_file(&p).ok();
				return Ok(None);
			}
		}
		if ext == "tmp" {
			return Ok(None);
		}
	}

	let file_type = dir_entry.file_type()?;

	// We allow the presence of directories in v1 (they represent namespace subdirectories)
	// and just skip them. In v2, directories at the key level are unexpected.
	if file_type.is_dir() {
		if allow_dirs {
			return Ok(None);
		}
		debug_assert!(
			false,
			"Failed to list keys at path {}: unexpected directory",
			PrintableString(p.to_str().unwrap_or_default())
		);
		let msg = format!(
			"Failed to list keys at path {}: unexpected directory",
			PrintableString(p.to_str().unwrap_or_default())
		);
		return Err(lightning::io::Error::new(lightning::io::ErrorKind::Other, msg));
	}

	// If we otherwise don't find a file at the given path something went wrong.
	if !file_type.is_file() {
		debug_assert!(
			false,
			"Failed to list keys at path {}: file couldn't be accessed.",
			PrintableString(p.to_str().unwrap_or_default())
		);
		let msg = format!(
			"Failed to list keys at path {}: file couldn't be accessed.",
			PrintableString(p.to_str().unwrap_or_default())
		);
		return Err(lightning::io::Error::new(lightning::io::ErrorKind::Other, msg));
	}

	match p.file_name().and_then(|n| n.to_str()) {
		Some(key) if is_valid_kvstore_str(key) => Ok(Some(key.to_string())),
		_ => Ok(None),
	}
}

/// Extracts a namespace name from a directory path.
/// In v2, maps [`EMPTY_NAMESPACE_DIR`] directories back to empty strings.
fn dir_to_namespace(path: &Path, use_empty_ns_dir: bool) -> Option<String> {
	let name = path.file_name()?.to_str()?;
	if use_empty_ns_dir && name == EMPTY_NAMESPACE_DIR {
		Some(String::new())
	} else if is_valid_kvstore_str(name) {
		Some(name.to_string())
	} else {
		None
	}
}

/// Lists all keys in the given directory.
///
/// When `retry_on_race` is true (used by FilesystemStore v1), the function retries directory
/// reads when entries disappear between listing and metadata access, and directories are
/// silently skipped (they represent namespace subdirectories). When false (used by
/// FilesystemStoreV2), disappearing entries are silently skipped and unexpected directories
/// produce an error.
fn do_list(prefixed_dest: &Path, retry_on_race: bool) -> lightning::io::Result<Vec<String>> {
	if !prefixed_dest.exists() {
		return Ok(Vec::new());
	}

	// In v1, directories can exist alongside key files (they represent namespace subdirectories),
	// which is also why races with disappearing entries are possible and need retries.
	// In v2, the directory structure is always exactly two levels deep, so directories at the
	// key level are unexpected.
	let allow_dirs = retry_on_race;

	let mut keys;
	let mut retries = if retry_on_race { LIST_DIR_CONSISTENCY_RETRIES } else { 0 };

	loop {
		keys = Vec::new();
		let mut needs_retry = false;

		for dir_entry in fs::read_dir(prefixed_dest)? {
			let dir_entry = dir_entry?;
			match entry_to_key(&dir_entry, allow_dirs) {
				Ok(Some(key)) => keys.push(key),
				Ok(None) => {},
				Err(e) => {
					if e.kind() == lightning::io::ErrorKind::NotFound && retries > 0 {
						retries -= 1;
						needs_retry = true;
						break;
					} else if retry_on_race {
						return Err(e);
					}
					// When not retrying (v2), silently skip errors
				},
			}
		}

		if !needs_retry {
			break;
		}
	}

	Ok(keys)
}
