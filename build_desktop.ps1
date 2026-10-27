param($workdir)

$py = "C:\Users\Blue Code technology\AppData\Roaming\Python\Python312\Scripts\pyinstaller.exe"
$spec = "C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence\eti_desktop.spec"

& $py $spec
Write-Output "Build complete"