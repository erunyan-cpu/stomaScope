# Navigate to repo root
Set-Location -Path ".."

# Clean old builds
Remove-Item -Recurse -Force .\dist, .\build, .\app.spec -ErrorAction Ignore

# PyInstaller build
pyinstaller app.py `
    --name "stomaScope" `
    --icon "assets/stomaScope.ico"
    --onedir `
    --noconsole `
    --clean `
    --paths "C:/Users/erunyan/AppData/Local/Programs/Python/Python311" `  # update path if not your laptop
    --collect-submodules torch `
    --collect-data torch `
    --collect-data customtkinter `
    --add-data "models/best_stomata_model.pth;models" `
    --hidden-import=torch.distributed `
    --hidden-import=torch.distributed.rpc `
    --hidden-import=segmentation_models_pytorch `
    --hidden-import=skimage `
    --hidden-import=pandas `
    --hidden-import=sklearn `
    --hidden-import=matplotlib.backends.backend_agg `
    --hidden-import=matplotlib.backends.backend_svg

