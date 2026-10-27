# PyInstaller spec file for ETI Desktop Application
# -*- mode: python -*-

a = Analysis(
    ['app.py'],
    binaries=[],
    datas=[
        ('eti', 'eti'),
    ],
    hiddenimports=[
        'fastapi',
        'starlette',
        'uvicorn',
        'pandas',
        'numpy',
        'plotly',
        'pytest',
        'httpx',
        'streamlit',
        'streamlit.components.v1',
    ],
    hookspath=[],
    hookinfos=[],
    license_bypass=True,
)

p = PYZ(a.pure_files)

b = BuildingAnalysis(
    a,
    [],
    name="ETI Desktop",
    icon=None,
    strip=False,
    upx=False,
    debug=False,
    runtime_tmpdir=None,
    scripts=None,
)

exe = Executable(
    script='app.py',
    base='Win32GUI',
    target_name='eti_desktop.exe',
    copyright='Egreed Technology',
    company_name='Egreed Technology',
)