# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['general_log_graph_plot.py'],
             pathex=['.dll'],
             binaries=[],
             datas=[],
             hiddenimports=['scipy.spatial.transform._rotation_groups','scipy.special.cython_special'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='general_log_graph_plot',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True , icon='.dll\\skate_icon_133135.ico')