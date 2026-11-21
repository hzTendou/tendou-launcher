use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, RunEvent};

const SERVER_BASE_URL: &str = "http://127.0.0.1:8000";
const API_V1_URL: &str = "http://127.0.0.1:8000/v1";

#[cfg(windows)]
struct WinJobGuard {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(windows)]
unsafe impl Send for WinJobGuard {}

#[cfg(windows)]
impl Drop for WinJobGuard {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

struct ManagedBackend {
    child: Option<Child>,
    #[cfg(windows)]
    _win_job: Option<WinJobGuard>,
}

struct BackendState(Mutex<ManagedBackend>);

fn is_server_healthy() -> bool {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(800))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client
        .get(format!("{}/health", SERVER_BASE_URL))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn find_project_root() -> Option<PathBuf> {
    if let Ok(env_root) = std::env::var("TENDOU_ROOT") {
        let p = PathBuf::from(env_root);
        if p.join("atlas_server.py").is_file() {
            return Some(p);
        }
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.to_path_buf());
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest_dir);

    for start in candidates {
        let mut cur: Option<&Path> = Some(start.as_path());
        while let Some(dir) = cur {
            if dir.join("atlas_server.py").is_file() {
                return Some(dir.to_path_buf());
            }
            cur = dir.parent();
        }
    }
    None
}

fn resolve_python_executable(root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        let venv_audit = root.join(".venv-audit").join("Scripts").join("python.exe");
        if venv_audit.is_file() {
            return venv_audit;
        }
        let venv = root.join(".venv").join("Scripts").join("python.exe");
        if venv.is_file() {
            return venv;
        }
        PathBuf::from("python.exe")
    }
    #[cfg(not(windows))]
    {
        let venv_audit = root.join(".venv-audit").join("bin").join("python");
        if venv_audit.is_file() {
            return venv_audit;
        }
        let venv = root.join(".venv").join("bin").join("python");
        if venv.is_file() {
            return venv;
        }
        PathBuf::from("python3")
    }
}

#[cfg(windows)]
fn attach_child_to_kill_on_close_job(child: &Child) -> Option<WinJobGuard> {
    use std::mem::size_of;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return None;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok_set = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const std::ffi::c_void,
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok_set == 0 {
            windows_sys::Win32::Foundation::CloseHandle(job);
            return None;
        }
        let proc_handle = child.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
        if AssignProcessToJobObject(job, proc_handle) == 0 {
            windows_sys::Win32::Foundation::CloseHandle(job);
            return None;
        }
        Some(WinJobGuard { handle: job })
    }
}

fn spawn_backend_if_needed() -> ManagedBackend {
    if is_server_healthy() {
        return ManagedBackend {
            child: None,
            #[cfg(windows)]
            _win_job: None,
        };
    }

    let Some(root) = find_project_root() else {
        eprintln!("[Tendou Studio] Could not locate atlas_server.py; waiting for external server.");
        return ManagedBackend {
            child: None,
            #[cfg(windows)]
            _win_job: None,
        };
    };

    let py = resolve_python_executable(&root);
    let server_script = root.join("atlas_server.py");

    let mut cmd = Command::new(&py);
    cmd.current_dir(&root)
        .arg(&server_script)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg("8000")
        .arg("--no-auto-start-engine")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.spawn() {
        Ok(child) => {
            #[cfg(windows)]
            let win_job = attach_child_to_kill_on_close_job(&child);
            // Wait briefly (up to ~2.5s) for FastAPI /health so WebView opens connected without an initial error flash
            for _ in 0..17 {
                if is_server_healthy() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(150));
            }
            ManagedBackend {
                child: Some(child),
                #[cfg(windows)]
                _win_job: win_job,
            }
        }
        Err(err) => {
            eprintln!("[Tendou Studio] Failed to spawn backend via {:?}: {}", py, err);
            ManagedBackend {
                child: None,
                #[cfg(windows)]
                _win_job: None,
            }
        }
    }
}

fn shutdown_managed_backend(app: &AppHandle) {
    if let Some(state) = app.try_state::<BackendState>() {
        if let Ok(mut guard) = state.0.lock() {
            if let Some(mut child) = guard.child.take() {
                // Request graceful VRAM release via Studio API first
                if let Ok(client) = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(2))
                    .build()
                {
                    let _ = client
                        .post(format!("{}/api/ui/engine/stop", SERVER_BASE_URL))
                        .send();
                }

                #[cfg(unix)]
                {
                    let pid = child.id() as libc::pid_t;
                    unsafe {
                        libc::kill(-pid, libc::SIGTERM);
                    }
                    let mut exited = false;
                    for _ in 0..15 {
                        if matches!(child.try_wait(), Ok(Some(_))) {
                            exited = true;
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    if !exited {
                        unsafe {
                            libc::kill(-pid, libc::SIGKILL);
                        }
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }

                #[cfg(windows)]
                {
                    let _ = child.kill();
                    let _ = child.wait();
                    guard._win_job = None; // Drops Win32 Job Object -> guarantees zero zombies
                }
            }
        }
    }
}

fn copy_text_to_clipboard(text: &str) {
    #[cfg(windows)]
    {
        if let Ok(mut proc) = Command::new("cmd")
            .args(["/C", "clip"])
            .stdin(Stdio::piped())
            .spawn()
        {
            if let Some(mut stdin) = proc.stdin.take() {
                use std::io::Write;
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = proc.wait();
        }
    }
    #[cfg(not(windows))]
    {
        for (bin, args) in [
            ("wl-copy", &[][..]),
            ("xclip", &["-selection", "clipboard"][..]),
            ("xsel", &["--clipboard", "--input"][..]),
        ] {
            if let Ok(mut proc) = Command::new(bin).args(args).stdin(Stdio::piped()).spawn() {
                if let Some(mut stdin) = proc.stdin.take() {
                    use std::io::Write;
                    let _ = stdin.write_all(text.as_bytes());
                }
                if proc.wait().map(|s| s.success()).unwrap_or(false) {
                    return;
                }
            }
        }
    }
}

fn show_main_window(app: &AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.unminimize();
        let _ = win.show();
        let _ = win.set_focus();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let managed = spawn_backend_if_needed();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .manage(BackendState(Mutex::new(managed)))
        .setup(|app| {
            let open_item = MenuItem::with_id(app, "open", "Open Tendou Studio", true, None::<&str>)?;
            let restart_item = MenuItem::with_id(app, "restart", "Restart Engine", true, None::<&str>)?;
            let copy_item = MenuItem::with_id(
                app,
                "copy_url",
                format!("Copy API URL ({})", API_V1_URL),
                true,
                None::<&str>,
            )?;
            let sep = PredefinedMenuItem::separator(app)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit & Stop Engine", true, None::<&str>)?;

            let tray_menu = Menu::with_items(app, &[&open_item, &restart_item, &copy_item, &sep, &quit_item])?;

            let mut tray_builder = TrayIconBuilder::new()
                .menu(&tray_menu)
                .tooltip("Tendou Studio v0.16.3 — Hybrid MoE Inference Engine (127.0.0.1:8000)")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show_main_window(app),
                    "restart" => {
                        std::thread::spawn(|| {
                            if let Ok(client) = reqwest::blocking::Client::builder()
                                .timeout(Duration::from_secs(30))
                                .build()
                            {
                                let _ = client
                                    .post(format!("{}/api/ui/engine/restart", SERVER_BASE_URL))
                                    .header("Content-Type", "application/json")
                                    .body("{}")
                                    .send();
                            }
                        });
                    }
                    "copy_url" => copy_text_to_clipboard(API_V1_URL),
                    "quit" => {
                        shutdown_managed_backend(app);
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_main_window(tray.app_handle());
                    }
                });

            if let Some(icon) = app.default_window_icon() {
                tray_builder = tray_builder.icon(icon.clone());
            }
            let _ = tray_builder.build(app)?;

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Tendou Studio Tauri application");

    app.run(|app_handle, event| {
        if let RunEvent::Exit = event {
            shutdown_managed_backend(app_handle);
        }
    });
}
