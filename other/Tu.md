
# 文本编辑: VS Code

Microsoft, Visual Stuido Code [Document](https://code.visualstudio.com/docs)

各种编辑器比较：
- Sublime Text, TextWrangler, Notepad++：好用，但是拓展性非常有限；
- Vim 和 Emacs：有充分定制的能力，但却有着陡峭的学习曲线；
- Atom：兼顾易用性和可拓展性，比较耗资源；
- VS 
Code：类似于Atom，但是节省资源，能够处理大文件，是目前实验室推荐编辑器。

**具体用途**
- 写普通文本文档
- 写程序
- 写 LabNotes：整合了 Markdown 和 Git，比以前 wiki 的 web editor 
容易写，容易推送
- 写 LaTeX 报告：整合了 LaTeX 和 
Git，不需要专门编辑器，快捷键编译，容易共享，容易跟踪改动

优点
- 资源占用比 Atom 小，能够轻松处理大文件
- 修改配置很方便，可以直接搜索
- Markdown: 可以点击链接，Outline模式很好用
- Git：默认支持
  - 日常操作非常方便，stage/commit/push/pull/sync/history，几乎不需要其他客户端、命令行界面；
  - Gutter indicators：有改动地方的色彩提示, red triangle/green bar/blue 
bar，点击可以显示与上次的差异；
  - View Diff: open changes 按钮, git history, git file history，git line 
history；
  - Merge：inline actions 按钮
- LaTeX：插件支持，可以自动编译，count word；
- Debug：默认支持，可以用于写 Python

## Settings

将主要配置文件（LabNotes\Pages\WritingTools\assets\keybindings.json,settings.json）拷贝到本地文件夹：
- for mac
    - ~/Library/Application\ Support/Code/User/
    - ~/Library/Application\ Support/Code/User/
- for windows
    - C:\Users\username\AppData\Roaming\Code\User\

Keyboard shortcuts:
| Key           | Action                    |
| :------------ | ------------------------- |
| Cmd-T         | Quick find/open files ⭐️  |
| Shift-Cmd-P   | Command Palette       ️⭐️  |
|               |                           |
| Shift-Cmd-[]  | Switch tab                |
| Shift-Cmd-E   | Explorer                  |
| Shift-Cmd-F   | Find in files             |
| Shift-Cmd-G   | Git                       |
| Shift-Cmd-X   | Extension                 |
| Cmd-\         | Toggle Sidebar            |
| Shift-Cmd-C   | Terminal here             |
| Shift-Cmd-\   | Markdown/LaTeX, Preview   |
|               |                           |
| Cmd-F         | Find                      |
| Cmd-R         | Replace                   |
| Cmd-[]        | In/outdent                |
| Cmd-/         | Toggle comment            |
| Cmd-B         | Markdown, bold            |
| Shift-Cmd-V   | Markdown, Paste image     |

## Extensions

Git:
- default support, commit/pull/push etc, **diff 模式很方便**
- Git History

Markdown:
- Markdown All in One: keyboard shortcut, list editing, table formatter, 
**outline**
- Markdown Preview Enhanced
- :emojisense:
- Paste Image
  - Ctrl-Alt-V
  - 不能贴文件，必须贴图本身
  - 修改：


```
code 
~/.vscode/extensions/mushan.vscode-paste-image-0.9.5/out/src/extension.js

imageFileName = moment().format("YMMDDHHmmss") + ".png";

//return "![](" + imageFilePath + ")";
return "<img src=\"" + imageFilePath + "\" width=500>";

// should also change settings, pasteImage.path
    "pasteImage.path": "${currentFileDir}/assets",
```

LaTeX:
- LaTeX Workshop
  - Build LaTeX (including BibTeX) to PDF **automatically on save**.
  - SyncTeX?
  - **Count word**
  - Note: change toolchain to 'xelatex'

Misc:
- Path Intellisense
- Atom Keymap
- vscode-icons
- File Utils

## Misc

**正则表达式，中文字符**： `[\u4e00-\u9fa5]`

**字体**：
- 编程字体的要求包括：
  * 等宽；
  * 支持语言覆盖面越大越好；
* 高的识别度（Legibility，非 Readability），能准确区分 Il1、O0o、各种标点符号等。在各种平台上清晰显示。
  - 对中国用户而言，还需要满足以下两个附加要求
    * 覆盖 GBK 全区段，当然汉字越多越好
    * 汉字宽度是西文严格的两倍，这样才能保证等宽
- [Inzui losevka](https://github.com/be5invis/Iosevka) Coders’ typeface, 
built from code. Inziu Iosevka is a composite of Iosevka, M+ and Source 
Han Sans.

**Emojis**:
| tag                | icon | tag         | icon |
|--------------------|------|-------------|------|
| :white_check_mark: | ✅   | :X:         | ❌   |
| :point_up:         | ☝️   | :pencil2:   | ✏️   |
| :sparkle:          | ❇️   | :hourglass: | ⌛   |
| :maple_leaf:       | 🍁   | :heart:     | ❤️   |
| :sunny:            | ☀️   | :bell:      | 🔔   |