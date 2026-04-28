# Prerelease Demo

This folder contains a reproducible one-minute terminal demo for the offline
compile-error autofix path.

The demo intentionally uses `UESPEC_BRIDGE_MODE=mock` and
`UESPEC_LLM_PROVIDER=mock`, so it does not require Unreal or a real LLM key. It
shows the public prerelease flow:

1. Install the package in editable mode.
2. Run the compile-error eval suite.
3. Print the generated report.

## Run Locally

PowerShell:

```powershell
.\docs\demo\prerelease-demo.ps1
```

Bash:

```bash
./docs/demo/prerelease-demo.sh
```

## Record on Windows

Use the built-in PowerShell transcript recorder. It does not require WSL,
asciinema, OBS, or any extra install:

```powershell
.\docs\demo\record-windows-demo.ps1
```

The script writes a text recording to:

```text
docs/demo/output/prerelease-demo-<timestamp>.transcript.txt
```

This is the default Windows-compatible terminal recording artifact. If a visual
GIF or MP4 is needed, record the same `prerelease-demo.ps1` run with ScreenToGif,
OBS, or another GUI recorder.

## Record With Asciinema

`asciinema` is optional and requires a Unix PTY. Use it only on Linux, macOS, or
WSL:

```bash
python -m pip install --user asciinema
asciinema rec docs/demo/prerelease-demo.cast -c "./docs/demo/prerelease-demo.sh" --overwrite
```

If a GIF is needed, convert the `.cast` with a local cast-to-gif tool.
